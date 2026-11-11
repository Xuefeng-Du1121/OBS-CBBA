"""Opt-in advection by archived HYCOM surface currents; no network on generation.

The only supported input is a small, rectilinear HYCOM NCSS NetCDF3 subset.
Missing cells, extrapolation, non-surface depths and non-SI currents fail closed.
This is real-data-driven synthetic buoy motion, not measured buoy trajectories.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.io import netcdf_file

EARTH_RADIUS_M = 6371000.
MAX_SAMPLE_BYTES = 3_000_000
HYCOM_ENDPOINT = 'https://ncss.hycom.org/thredds/ncss/GLBy0.08/expt_93.0'
HYCOM_VERSION = 'GOFS 3.1 GLBy0.08/expt_93.0'
HYCOM_LICENSE = 'DoD DISTRIBUTION A. Approved for Public Release; Distribution Unlimited.'
HYCOM_LICENSE_URL = 'https://www.hycom.org/dataserver'


def utc_seconds(value):
    """Require an explicit timezone, so local timezone cannot alter a scenario."""
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Time must include a timezone (prefer UTC Z)')
    return dt.timestamp()


def utc_string(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace('+00:00', 'Z')


def file_sha256(path):
    path = Path(path)
    if path.stat().st_size > MAX_SAMPLE_BYTES:
        raise ValueError('Current subset exceeds the 3 MB input limit')
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class RealDriftParameters:
    file: str
    sha256: str
    center_lat: float
    center_lon: float
    start_utc: str
    area_m: float = 10000.
    dt_seconds: float = 25.

    def validate(self):
        if not isinstance(self.file, str) or not self.file:
            raise ValueError('A local NetCDF subset is required')
        if not re.fullmatch('[0-9a-f]{64}', self.sha256):
            raise ValueError('An explicit SHA256 pin is required')
        if not np.isfinite([self.center_lat, self.center_lon, self.area_m, self.dt_seconds]).all():
            raise ValueError('Nonfinite real drift parameter')
        if not (-75 <= self.center_lat <= 75 and -180 <= self.center_lon < 180):
            raise ValueError('Use longitude [-180,180) and latitude [-75,75]')
        if self.area_m != 10000. or self.dt_seconds != 25.:
            raise ValueError('This adapter uses the fixed 10 km / 25 s physical mapping')
        utc_seconds(self.start_utc)
        return self

    @classmethod
    def from_config(cls, cfg):
        if cfg.mobility != 'real-current' or not isinstance(cfg.real_drift, dict):
            raise ValueError('Explicit real-current parameters required')
        if cfg.sigma != 0 or cfg.nominal_speed != 0 or cfg.maritime is not None:
            raise ValueError('Real currents require zero legacy motion and no synthetic maritime forcing')
        return cls(**cfg.real_drift).validate()

    def bounds(self):
        """Local equirectangular square; x east, y north, origin southwest."""
        dy = np.degrees(self.area_m / (2 * EARTH_RADIUS_M))
        dx = dy / np.cos(np.radians(self.center_lat))
        return dict(west=self.center_lon-dx, east=self.center_lon+dx,
                    south=self.center_lat-dy, north=self.center_lat+dy)


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _bracket(axis, low, high):
    if len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError('Axes need at least two finite, strictly increasing coordinates')
    if low < axis[0] or high > axis[-1] or low > high:
        raise ValueError('Requested window is outside available current coverage')
    first = max(0, np.searchsorted(axis, low, side='right') - 1)
    last = min(len(axis), np.searchsorted(axis, high, side='left') + 1)
    return slice(first, max(first + 2, last))


@dataclass
class CurrentField:
    times: np.ndarray  # UTC Unix seconds
    lat: np.ndarray
    lon: np.ndarray  # unwrapped around the requested center
    velocity: np.ndarray  # time, lat, lon, (east,north), m/s

    def select(self, p, steps):
        if not isinstance(steps, (int, np.integer)) or steps < 2:
            raise ValueError('At least two position frames required')
        b = p.validate().bounds()
        start = utc_seconds(p.start_utc)
        ts = _bracket(self.times, start, start + (steps-1)*p.dt_seconds)
        ys = _bracket(self.lat, b['south'], b['north'])
        xs = _bracket(self.lon, b['west'], b['east'])
        if self.velocity.shape != (len(self.times), len(self.lat), len(self.lon), 2):
            raise ValueError('Velocity dimensions must be time, latitude, longitude, 2')
        selected = CurrentField(self.times[ts], self.lat[ys], self.lon[xs], self.velocity[ts, ys, xs])
        if not np.isfinite(selected.velocity).all():
            raise ValueError('Selected window contains missing/land current cells')
        return selected


def load_hycom(path, center_lon, expected_sha256):
    """Parse untrusted data with SciPy, never pickle or execute downloaded files."""
    if file_sha256(path) != expected_sha256:
        raise ValueError('Current subset SHA256 mismatch')
    with netcdf_file(path, 'r', mmap=False, maskandscale=False) as nc:
        def axis(name, units):
            v = nc.variables[name]
            if v.dimensions != (name,) or _text(v.units) != units:
                raise ValueError('Unsupported coordinate units or dimensions: ' + name)
            return v[:].astype(float).copy()
        lat = axis('lat', 'degrees_north')
        lon = axis('lon', 'degrees_east')
        depth = axis('depth', 'm')
        if depth.shape != (1,) or depth[0] != 0:
            raise ValueError('Request exactly the surface depth (0 m)')
        tv = nc.variables['time']
        if tv.dimensions != ('time',):
            raise ValueError('Unsupported time dimensions')
        if _text(getattr(tv, 'calendar', b'standard')) not in ('standard', 'gregorian', 'proleptic_gregorian'):
            raise ValueError('Unsupported current calendar')
        match = re.fullmatch(r'(seconds|hours|days) since (.+)', _text(tv.units))
        if not match:
            raise ValueError('Unsupported current time units')
        epoch = match[2].replace(' UTC', '+00:00')
        if datetime.fromisoformat(epoch.replace('Z', '+00:00')).tzinfo is None:
            epoch += '+00:00'  # CF reference times without zones mean UTC.
        times = np.asarray(tv[:], dtype=float) * {'seconds': 1, 'hours': 3600, 'days': 86400}[match[1]] + utc_seconds(epoch)
        components = []
        for name in ('water_u', 'water_v'):
            var = nc.variables[name]
            if var.dimensions != ('time', 'depth', 'lat', 'lon') or _text(var.units) != 'm/s':
                raise ValueError('Unsupported velocity units or dimensions')
            raw = var[:, 0, :, :].astype(float).copy()
            missing = ~np.isfinite(raw)
            for attr in ('_FillValue', 'missing_value'):
                if hasattr(var, attr):
                    missing |= np.isin(raw, getattr(var, attr))
            values = raw * float(getattr(var, 'scale_factor', 1.)) + float(getattr(var, 'add_offset', 0.))
            values[missing] = np.nan
            components.append(values)
    lon = (lon - center_lon + 180.) % 360. + center_lon - 180.
    ti, yi, xi = np.argsort(times), np.argsort(lat), np.argsort(lon)
    velocity = np.stack(components, axis=-1)[ti][:, yi][:, :, xi]
    return CurrentField(times[ti], lat[yi], lon[xi], velocity)


def advect_positions(field, initial, release, steps, p):
    """Forward Euler, bilinear space/linear time; mirror proposals into [0,1]."""
    field = field.select(p, steps)
    initial = np.asarray(initial, dtype=float)
    release = np.asarray(release)
    if initial.ndim != 2 or initial.shape[1] != 2 or release.shape != (len(initial),):
        raise ValueError('Initial positions must be tasks x 2, with one release per task')
    if (not np.isfinite(initial).all() or np.any((initial < 0) | (initial > 1))
            or not np.issubdtype(release.dtype, np.integer) or np.any(release < 0)):
        raise ValueError('Positions must lie in [0,1]; releases must be nonnegative integer ticks')
    interpolation = RegularGridInterpolator((field.times, field.lat, field.lon), field.velocity,
                                            bounds_error=True)
    b = p.bounds(); start = utc_seconds(p.start_utc)
    positions = np.empty((steps, len(initial), 2), dtype=np.float64)
    positions[0] = initial
    for t in range(1, steps):
        old = positions[t-1]
        lat = b['south'] + old[:, 1] * (b['north']-b['south'])
        lon = b['west'] + old[:, 0] * (b['east']-b['west'])
        queries = np.column_stack((np.full(len(initial), start+(t-1)*p.dt_seconds), lat, lon))
        moved = old + interpolation(queries) * p.dt_seconds / p.area_m
        reflected = 1. - np.abs(np.mod(moved, 2.) - 1.)
        positions[t] = np.where((release < t)[:, None], reflected, initial)
    return positions


def generate_real_drift(cfg, seed):
    from .arrivals import release_schedule, deadlines
    from .scenario import Scenario
    if 40000 <= seed <= 40999:
        raise ValueError('Sealed confirmation seeds are not allowed in this preparation adapter')
    p = RealDriftParameters.from_config(cfg)
    field = load_hycom(p.file, p.center_lon, p.sha256)
    initial_rng, _, comm_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    arrival_rng = np.random.default_rng(np.random.SeedSequence([0x41525231, seed]))
    release = release_schedule(cfg, arrival_rng)
    steps = cfg.horizon + cfg.return_steps + 1
    initial = initial_rng.uniform(0, 1, (len(release), 2))
    positions = advect_positions(field, initial, release, steps, p)
    uniform = comm_rng.random((steps, len(release), cfg.uavs)).astype(np.float32)
    return Scenario(positions, release, deadlines(cfg, release), uniform, int(seed), cfg.dict())


def processing_provenance(p, steps):
    """JSON-safe processing record; caller adds source URL, hash, license, retrieval time."""
    return dict(implementation='dhgc.real_drift/v1', space_window=p.bounds(),
                time_window=dict(start=utc_string(utc_seconds(p.start_utc)),
                                 end=utc_string(utc_seconds(p.start_utc)+(steps-1)*p.dt_seconds)),
                area_m=p.area_m, dt_seconds=p.dt_seconds, axes='x=east, y=north',
                projection='local equirectangular, Earth radius 6371000 m, center-latitude cosine',
                integrator='forward Euler; bilinear spatial and linear temporal interpolation',
                boundary='reflect each proposal: 1-abs(mod(x,2)-1); artificial square, not coastline',
                release='stationary through release tick r; first displacement at frame r+1',
                forcing='surface current only; no added windage, waves, residual or mooring',
                missing_data='reject selected land/missing cells and all space/time extrapolation')
