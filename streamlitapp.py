# -*- coding: utf-8 -*-
"""V2.1 Streamlit application with a bundled fitted model.

Place V2.1_AB_models.joblib, requirements.txt and packages.txt beside this
file in the repository. Select Python 3.12 when deploying to Streamlit.
Run locally: python -m streamlit run streamlitapp.py
"""
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
import csv
import hashlib
import io
import math
import os
import sys
import types

sys.dont_write_bytecode = True
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

MODEL_PATH = Path(__file__).resolve().with_name('V2.1_AB_models.joblib')


MODEL_PACKAGE_SHA256 = 'e21c6a491899a9fbf443337a4c7428697409d1e050ffcc24025e7d4a7ce20354'

STATIC = ['D/t', 'L/D', 'Helix Angle', 'fcuAc', 'fyAs', 'h/l', 'Concrete Types', 'e/r', 'Confinement Factor']

MODEL_INFO = {'A': {'sha256': '71b80bd12b8c00791175c80d0b16ba16bb01891a4a1acde4419602e9e3b6dc73', 'q90': 0.2457351539372437, 'ranges': {'D/t': [8.70945557167, 628.5], 'L/D': [1.19, 18.0], 'Helix Angle': [0.0, 42.0], 'fcuAc': [156573.914787, 65696825.3691], 'fyAs': [64204.5119552, 10491412.8968], 'h/l': [0.0, 0.33], 'Concrete Types': [0.0, 0.2921378609929999], 'e/r': [0.0, 1.35], 'Confinement Factor': [0.0471229493788, 4.47647101365]}, 'defaults': {'D/t': 53.9788617812, 'L/D': 3.02023556161, 'Helix Angle': 0.0, 'fcuAc': 899993.816015, 'fyAs': 512412.184756, 'h/l': 0.0, 'Concrete Types': 0.0, 'e/r': 0.0, 'Confinement Factor': 0.569350784014}}, 'B': {'sha256': 'e37f328b92681c4c597e7e6b765b075be8c02c4b3cb904b9d4c33357a63937a0', 'q90': 0.2696798845933755, 'ranges': {'D/t': [8.70945557167, 298.669623103], 'L/D': [1.88810680588, 18.0], 'Helix Angle': [0.0, 39.1145279017], 'fcuAc': [126679.835236, 17830916.9753], 'fyAs': [64204.5119552, 10491412.8968], 'h/l': [0.0, 0.209271420536], 'Concrete Types': [0.0, 0.599205434624], 'e/r': [0.0, 1.22724799376], 'Confinement Factor': [0.0683108455347, 4.47647101365]}, 'defaults': {'D/t': 178.284726156, 'L/D': 4.40823440171, 'Helix Angle': 0.0, 'fcuAc': 1908534.38938, 'fyAs': 547214.166156, 'h/l': 0.191277275733, 'Concrete Types': 0.209028212594, 'e/r': 0.0, 'Confinement Factor': 0.286719573512}}}

_COMPAT_SOURCES = {'revision_model': "@dataclass\nclass Config:\n    random_state: int = 42\n    gate_folds: int = 5\n    search_folds: int = 3\n    search_iterations: int = 12\n    constraint_mode: str = 'limited'\n    physics_form: str = 'additive'\n    ridge_alpha: float = 1.0\n    refit_full_development: bool = True\n    smoke: bool = False\n\nclass PhysicsBaseline:\n\n    def __init__(self, form='additive', unit_scale=1000.0):\n        self.form = form\n        self.unit_scale = float(unit_scale)\n\n    def transform(self, X):\n        z = X[RAW].astype(float).copy()\n        z['N0_Enhanced'] = (z['fcuAc'] * (1 + self.alpha_ * z['Confinement Factor']) + z['fyAs']) / self.unit_scale\n        z['PhysicsReduction'] = 1 / (1 + self.beta_ * z['e/r'] + self.gamma_ * z['L/D']) if self.form == 'additive' else 1 / ((1 + self.beta_ * z['e/r']) * (1 + self.gamma_ * z['L/D']))\n        z['N_phy'] = z['N0_Enhanced'] * z['PhysicsReduction']\n        z['BoundaryFactor'] = (np.abs(z['Strain']) > 1e-12).astype(float)\n        if not (z['N_phy'] > 0).all():\n            raise ValueError('Non-positive physics scale.')\n        return z[FEATURES]\n\nclass PhysicsResidualRegressor(RegressorMixin, BaseEstimator):\n\n    def __init__(self, estimator=None, physics_form='additive'):\n        self.estimator = estimator\n        self.physics_form = physics_form\n\n    def predict(self, X):\n        z = self.physics_.transform(X)\n        ratio = np.maximum(self.estimator_.predict(z), 0.0)\n        return z['BoundaryFactor'].to_numpy() * z['N_phy'].to_numpy() * ratio\n\nclass UnanchoredStack:\n\n    def __init__(self, config):\n        self.config = config\n\n    def predict_parts(self, X):\n        z = self.physics_.transform(X).to_numpy()\n        raw_errors = np.column_stack([g.predict(z) for g in self.gates_])\n        clipped_errors = np.maximum(raw_errors, 0.0)\n        weights = softmax(-clipped_errors / self.temperature_, axis=1)\n        base = np.column_stack([m.predict(X) for m in self.base_models_])\n        return (np.sum(weights * base, axis=1), base, weights, raw_errors)\n\nclass RevisionEnsemble:\n\n    def __init__(self, config):\n        self.config = config\n\n    def predict(self, X, return_details=False):\n        dgs, base, weights, errors = self.stack_.predict_parts(X)\n        a = self.anchor_alpha_\n        result = a * dgs + (1 - a) * base[:, 2]\n        effective = a * weights\n        effective[:, 2] += 1 - a\n        return (result, weights, effective, errors) if return_details else result", 'formal_engine': "class FirstTen(TransformerMixin, BaseEstimator):\n\n    def transform(self, X):\n        return np.asarray(X)[:, :10]\n\nclass ConstantScale:\n\n    def transform(self, X):\n        z = X[RAW].astype(float).copy()\n        z['N0_Enhanced'] = self.scale_\n        z['PhysicsReduction'] = 1.0\n        z['N_phy'] = self.scale_\n        z['BoundaryFactor'] = (z.Strain.abs() > 1e-12).astype(float)\n        return z[FEATURES]\n\nclass DirectCapacity(RegressorMixin, BaseEstimator):\n\n    def __init__(self, estimator=None, physics_form='additive', feature_mode='physics'):\n        self.estimator = estimator\n        self.physics_form = physics_form\n        self.feature_mode = feature_mode\n\n    def predict(self, X):\n        z = self.physics_.transform(X)\n        xx = z if self.feature_mode == 'physics' else z[RAW]\n        return z.BoundaryFactor.to_numpy() * np.maximum(self.estimator_.predict(xx), 0)\n\nclass CachedStack(ORIGINAL_STACK):\n    pass", 'optimization_engine': "class WeightedResidual(RegressorMixin, BaseEstimator):\n\n    def __init__(self, estimator=None, physics_form='additive', group_exponent=1.0, peak_coefficient=0.0):\n        self.estimator = estimator\n        self.physics_form = physics_form\n        self.group_exponent = group_exponent\n        self.peak_coefficient = peak_coefficient\n\n    def predict(self, X):\n        z = self.physics_.transform(X)\n        return z.BoundaryFactor.to_numpy() * z.N_phy.to_numpy() * np.maximum(self.estimator_.predict(z), 0.0)\n\nclass WeightedStack:\n\n    def __init__(self, group_exponent=1.0, peak_coefficient=0.0):\n        self.group_exponent = group_exponent\n        self.peak_coefficient = peak_coefficient\n\n    def predict_parts(self, X):\n        z = self.gate_scaler_.transform(self.physics_.transform(X).to_numpy())\n        errors = np.column_stack([g.predict(z) for g in self.gates_])\n        weights = softmax(-np.maximum(errors, 0) / self.temperature_, axis=1)\n        base = np.column_stack([m.predict(X) for m in self.base_models_])\n        return (np.sum(weights * base, axis=1), base, weights, errors)\n\nclass CurveModel:\n\n    def __init__(self, stack, alpha):\n        self.stack_ = stack\n        self.anchor_alpha_ = float(alpha)\n\n    def predict(self, X, return_details=False):\n        dgs, base, w, e = self.stack_.predict_parts(X)\n        a = self.anchor_alpha_\n        pred = a * dgs + (1 - a) * base[:, 2]\n        effective = a * w\n        effective[:, 2] += 1 - a\n        return (pred, w, effective, e) if return_details else pred", 'capacity_branch': "class CapacityResidual(RegressorMixin, BaseEstimator):\n\n    def __init__(self, estimator=None, group_exponent=1.0):\n        self.estimator = estimator\n        self.group_exponent = group_exponent\n\n    def predict(self, X):\n        scale = self.physics_.transform(X).N_phy.to_numpy()\n        residual = self.estimator_.predict(X[STATIC])\n        result = scale * np.exp(residual)\n        if not np.isfinite(result).all():\n            raise ValueError('Nonfinite capacity prediction.')\n        return result\n\nclass CapacityCurveModel:\n\n    def __init__(self, shape_model, capacity_models, capacity_weights):\n        self.shape_model_ = shape_model\n        self.capacity_models_ = capacity_models\n        self.capacity_weights_ = np.asarray(capacity_weights, float)\n        assert (self.capacity_weights_ >= 0).all() and abs(self.capacity_weights_.sum() - 1) < 1e-08\n\n    def predict_capacity(self, X):\n        return np.column_stack([m.predict(X) for m in self.capacity_models_]) @ self.capacity_weights_\n\n    def predict(self, X):\n        X = X.reset_index(drop=True)\n        pred = np.empty(len(X))\n        raw = self.shape_model_.predict(X)\n        for sid, pos in X.groupby('Specimen ID', sort=False).indices.items():\n            pos = np.asarray(pos)\n            s = X.iloc[pos].Strain.to_numpy()\n            if len(pos) < 3 or abs(s.min()) > 1e-12 or abs(s.max() - 0.03) > 1e-12:\n                raise ValueError('predict requires complete curves covering strain 0 to 0.03; use predict_at_points with an explicit reference grid for point requests.')\n            assert X.iloc[pos][STATIC].nunique().max() == 1\n            capacity = float(self.predict_capacity(X.iloc[pos[:1]])[0])\n            maximum = float(np.max(raw[pos]))\n            if maximum <= 0:\n                raise ValueError('Shape branch has no positive full-curve prediction.')\n            pred[pos] = capacity * np.maximum(raw[pos], 0) / maximum\n        return pred\n\n    def predict_at_points(self, X, reference_grids=None):\n        X = X.reset_index(drop=True)\n        answer = np.empty(len(X))\n        for sid, pos in X.groupby('Specimen ID', sort=False).indices.items():\n            pos = np.asarray(pos)\n            grid = np.linspace(0, 0.03, 101) if reference_grids is None else np.asarray(reference_grids[int(sid)], float)\n            assert grid[0] == 0 and abs(grid[-1] - 0.03) < 1e-12 and (np.diff(grid) > 0).all()\n            row = X.iloc[pos[0]].to_dict()\n            full = pd.DataFrame([row] * len(grid))\n            full['Strain'] = grid\n            values = self.predict(full)\n            query = X.iloc[pos].Strain.to_numpy()\n            if (query < 0).any() or (query > 0.03).any():\n                raise ValueError('Requested strain outside reference curve.')\n            answer[pos] = np.interp(query, grid, values)\n        return answer", 'scale_calibration': 'class ScaledCapacityCurve(CapacityCurveModel):\n\n    def __init__(self, base_model, capacity_scale=1.0):\n        super().__init__(base_model.shape_model_, base_model.capacity_models_, base_model.capacity_weights_)\n        self.capacity_scale_ = float(capacity_scale)\n        assert self.capacity_scale_ > 0\n\n    def predict_capacity(self, X):\n        return self.capacity_scale_ * super().predict_capacity(X)', 'pruned_shape': 'class SingleResidualCurve:\n\n    def __init__(self, base_model):\n        self.base_model_ = base_model\n        self.anchor_alpha_ = 0.0\n\n    def predict(self, X):\n        return self.base_model_.predict(X)'}

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def _register_compatibility_classes():
    """Register the original module/class names embedded in the joblib files."""
    import numpy as np
    import pandas as pd
    from dataclasses import dataclass
    from scipy.special import softmax
    from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
    shared = {'np': np, 'pd': pd, 'dataclass': dataclass, 'softmax': softmax, 'BaseEstimator': BaseEstimator, 'RegressorMixin': RegressorMixin, 'TransformerMixin': TransformerMixin, 'RAW': STATIC + ['Strain'], 'FEATURES': STATIC + ['Strain', 'N0_Enhanced', 'PhysicsReduction', 'N_phy', 'BoundaryFactor'], 'STATIC': STATIC}
    modules = {name: types.ModuleType(name) for name in _COMPAT_SOURCES}
    for name, module in modules.items():
        module.__dict__.update(shared)
        sys.modules[name] = module
    for name in ['revision_model', 'formal_engine', 'optimization_engine', 'capacity_branch', 'scale_calibration', 'pruned_shape']:
        module = modules[name]
        module.rm = modules['revision_model']
        if name == 'formal_engine':
            module.ORIGINAL_STACK = modules['revision_model'].UnanchoredStack
        if name == 'scale_calibration':
            module.CapacityCurveModel = modules['capacity_branch'].CapacityCurveModel
        exec(compile(_COMPAT_SOURCES[name], '<V2.1 compatibility: ' + name + '>', 'exec'), module.__dict__)

class ModelBundle:
    """Load the fitted A/B models once and serialize concurrent predictions."""

    def __init__(self, path=MODEL_PATH):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                'Place V2.1_AB_models.joblib in the same repository folder as streamlitapp.py.'
            )
        if _sha(path) != MODEL_PACKAGE_SHA256:
            raise ValueError('The model file does not match this V2.1 application. Upload the original file unchanged.')
        _register_compatibility_classes()
        import joblib
        package = joblib.load(path)
        if not isinstance(package, dict) or package.get('format') != 'V21_AB_FITTED_MODEL_PACKAGE':
            raise ValueError('Invalid fitted-model package format.')
        if package.get('version') != 'V2.1' or set(package.get('models', {})) != {'A', 'B'}:
            raise ValueError('The model package must contain both V2.1 A/B models.')
        if package.get('model_info') != MODEL_INFO:
            raise ValueError('The package parameters do not match this application version.')
        self.models = package['models']
        self._lock = RLock()

    def predict(self, scheme, features, start=0.0, end=0.03, count=101):
        if scheme not in self.models:
            raise ValueError('Select Scheme A or Scheme B.')
        with self._lock:
            return predict_curve(self.models[scheme], scheme, features, start, end, count)


def validate_features(features):
    missing = set(STATIC) - set(features)
    if missing:
        raise ValueError('Missing model inputs: ' + ', '.join(sorted(missing)))
    try:
        values = {key: float(features[key]) for key in STATIC}
    except (TypeError, ValueError) as exc:
        raise ValueError('Every model input must be a finite number.') from exc
    if not all((math.isfinite(value) for value in values.values())):
        raise ValueError('Every model input must be a finite number.')
    if values['D/t'] <= 2 or values['L/D'] <= 0:
        raise ValueError('D/t must exceed 2 and L/D must be positive.')
    if min(values['fcuAc'], values['fyAs']) <= 0:
        raise ValueError('fcuAc and fyAs must be positive (unit: N).')
    if not 0 <= values['Helix Angle'] <= 90:
        raise ValueError('Helix angle must be between 0 and 90 degrees.')
    if not 0 <= values['Concrete Types'] <= 1:
        raise ValueError('Rubber replacement ratio must be between 0 and 1.')
    if any((values[key] < 0 for key in ['h/l', 'e/r', 'Confinement Factor'])):
        raise ValueError('h/l, e/r and confinement factor cannot be negative.')
    return values

def validate_window(start, end, count):
    if not all((math.isfinite(float(v)) for v in [start, end, count])):
        raise ValueError('Strain limits and point count must be finite.')
    if not 0 <= start < end <= 0.03:
        raise ValueError('Select 0 ≤ start < end ≤ 0.03.')
    if isinstance(count, bool) or int(count) != count or (not 2 <= int(count) <= 1000):
        raise ValueError('The displayed curve must contain 2 to 1000 points.')

def geometry_to_features(D, t, L, h, spacing, fcu, fy, rubber_ratio, e_over_r, helix_angle, effective_areas=None):
    if not all((math.isfinite(float(v)) for v in [D, t, L, h, spacing, fcu, fy, rubber_ratio, e_over_r, helix_angle])):
        raise ValueError('All geometry and material inputs must be finite.')
    if min(D, t, L, spacing, fcu, fy) <= 0 or t >= D / 2 or h < 0:
        raise ValueError('Use positive dimensions and strengths, h ≥ 0 and t < D/2.')
    if effective_areas is None:
        Ac, As = (math.pi * (D - 2 * t) ** 2 / 4, math.pi * (D ** 2 - (D - 2 * t) ** 2) / 4)
    else:
        Ac, As = map(float, effective_areas)
        if not all((math.isfinite(v) and v > 0 for v in [Ac, As])):
            raise ValueError('Both effective areas must be finite and positive.')
    fcuAc, fyAs = (fcu * Ac, fy * As)
    values = validate_features({'D/t': D / t, 'L/D': L / D, 'Helix Angle': helix_angle, 'fcuAc': fcuAc, 'fyAs': fyAs, 'h/l': h / spacing, 'Concrete Types': rubber_ratio, 'e/r': e_over_r, 'Confinement Factor': fyAs / fcuAc})
    return (values, {'Ac_mm2': Ac, 'As_mm2': As})

def range_checks(scheme, features):
    return [{'Input': key, 'Value': value, 'Min': MODEL_INFO[scheme]['ranges'][key][0], 'Max': MODEL_INFO[scheme]['ranges'][key][1]} for key, value in features.items() if not MODEL_INFO[scheme]['ranges'][key][0] - 1e-09 <= value <= MODEL_INFO[scheme]['ranges'][key][1] + 1e-09]

def to_csv(records):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode('utf-8-sig')

def predict_curve(model, scheme, features, start=0.0, end=0.03, count=101):
    import numpy as np
    import pandas as pd
    values = validate_features(features)
    validate_window(start, end, count)
    grid = np.linspace(0.0, 0.03, 101)
    full = pd.DataFrame({key: np.repeat(value, 101) for key, value in values.items()})
    full['Strain'], full['Specimen ID'], full['SourceGroup'] = (grid, 0, 'APP_INPUT')
    prediction = np.asarray(model.predict(full), float)
    capacity = float(model.predict_capacity(full.iloc[[0]])[0])
    if prediction.shape != grid.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('The model returned an invalid curve.')
    if prediction[0] != 0 or not np.isclose(prediction.max(), capacity, rtol=1e-10, atol=1e-07):
        raise ValueError('The curve and capacity branch are inconsistent.')
    q = MODEL_INFO[scheme]['q90']
    lower, upper = (np.maximum(prediction - q * capacity, 0), prediction + q * capacity)
    lower[0], upper[0] = (0.0, 0.0)
    query = np.linspace(float(start), float(end), int(count))

    def records(x):
        p, lo, hi = [np.interp(x, grid, y) for y in [prediction, lower, upper]]
        return [{'Strain': float(a), 'Predicted_load_kN': float(b), 'Lower_kN': float(c), 'Upper_kN': float(d), 'Model_version': 'V2.1', 'Scheme': scheme, 'Full_curve_peak_kN': capacity, 'q90': q, **values} for a, b, c, d in zip(x, p, lo, hi)]
    displayed = records(query)
    return {'display': displayed, 'full': records(grid), 'features': values, 'protocol': scheme, 'capacity_kN': capacity, 'peak_strain': float(grid[np.argmax(prediction)]), 'window_peak_kN': max((row['Predicted_load_kN'] for row in displayed)), 'q90': q}

# Streamlit interface
from html import escape
import streamlit as st
st.set_page_config(page_title='CFST Load–Strain Prediction | V2.1', page_icon='🏗️', layout='wide')

def clear_prediction():
    st.session_state.pop('prediction', None)

@st.cache_resource(show_spinner=False)
def get_model():
    return ModelBundle()

@st.cache_data(show_spinner=False, max_entries=32)
def run_prediction(scheme, features, start, end, count):
    return get_model().predict(scheme, features, start, end, count)

def curve_svg(result, show_band):
    """Export the displayed curve without extra plotting dependencies."""
    rows = result['display']
    width, height, left, right, top, bottom = (800, 480, 75, 25, 45, 65)
    xmin, xmax = (rows[0]['Strain'], rows[-1]['Strain'])
    ymax = max((row['Upper_kN'] if show_band else row['Predicted_load_kN'] for row in rows)) * 1.08
    ymax = max(ymax, 1.0)

    def xy(row, field):
        x = left + (row['Strain'] - xmin) / (xmax - xmin) * (width - left - right)
        y = height - bottom - row[field] / ymax * (height - top - bottom)
        return f'{x:.3f},{y:.3f}'
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial,sans-serif" font-size="13">']
    for i in range(5):
        y = height - bottom - i / 4 * (height - top - bottom)
        x = left + i / 4 * (width - left - right)
        parts.append(f'<path d="M{left},{y} H{width - right}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{ymax * i / 4:.0f}</text>')
        parts.append(f'<text x="{x}" y="{height - bottom + 23}" text-anchor="middle">{xmin + (xmax - xmin) * i / 4:.4f}</text>')
    if show_band:
        polygon = ' '.join([xy(row, 'Upper_kN') for row in rows] + [xy(row, 'Lower_kN') for row in reversed(rows)])
        parts.append(f'<polygon points="{polygon}" fill="#D8E4F0" fill-opacity="0.65"/>')
    points = ' '.join((xy(row, 'Predicted_load_kN') for row in rows))
    parts.append(f'<polyline points="{points}" fill="none" stroke="#E07A5F" stroke-width="2"/>')
    parts.append(f'<path d="M{left},{top} V{height - bottom} H{width - right}" fill="none" stroke="#222"/>')
    parts.append(f'<text x="400" y="25" text-anchor="middle" font-size="17">{escape('V2.1 — Scheme ' + result['protocol'])}</text>')
    parts.append('<text x="420" y="465" text-anchor="middle">Strain</text>')
    parts.append('<text x="20" y="230" transform="rotate(-90 20 230)" text-anchor="middle">Load (kN)</text>')
    parts.append('</g></svg>')
    return ''.join(parts).encode('utf-8')
st.title('CFST Load–Strain Prediction')
st.caption('MultiPhysics-DGS-CA · V2.1')
with st.sidebar:
    st.header('Prediction settings')
    scheme = st.selectbox('Model scheme', ['A', 'B'], key='scheme', on_change=clear_prediction)
    mode = st.radio('Input mode', ['Geometry and materials', 'Model features'], key='input_mode', on_change=clear_prediction)
    if mode == 'Geometry and materials':
        st.subheader('Input conventions')
        area_mode = st.selectbox('Section areas', ['Circular section', 'Enter effective areas'], key='areas', on_change=clear_prediction)
        eccentricity_mode = st.selectbox('Eccentricity input', ['e/r directly', 'e in mm'], key='ecc_mode', on_change=clear_prediction)
        angle_mode = st.selectbox('Helix input', ['Angle in degrees', 'Wave number K'], key='angle_mode', on_change=clear_prediction)
    st.caption('Click Predict curve after changing the inputs.')
with st.form('prediction_form'):
    st.subheader('Model parameter input')
    features, geometry = (None, None)
    if mode == 'Geometry and materials':
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('**Geometry**')
            D = st.number_input('D — outer diameter (mm)', min_value=1.0, value=315.0, format='%.3f', key='diameter')
            t = st.number_input('t — wall thickness (mm)', min_value=0.001, value=2.0, format='%.3f', key='thickness')
            L = st.number_input('L — length (mm)', min_value=1.0, value=750.0, format='%.3f', key='length')
            h = st.number_input('h — corrugation height (mm)', min_value=0.0, value=0.0, format='%.3f', key='height')
            spacing = st.number_input('l — corrugation spacing (mm)', min_value=0.001, value=75.0, format='%.3f', key='spacing')
        with c2:
            st.markdown('**Materials**')
            fcu = st.number_input('fcu — concrete strength (MPa)', min_value=0.001, value=40.0, format='%.3f', key='fcu')
            fy = st.number_input('fy — steel yield strength (MPa)', min_value=0.001, value=235.0, format='%.3f', key='fy')
            rubber = st.number_input('Rubber replacement ratio', min_value=0.0, max_value=1.0, value=0.0, format='%.4f', key='rubber')
            if area_mode == 'Enter effective areas':
                Ac = st.number_input('Effective concrete area Ac (mm²)', min_value=0.001, value=75963.0, format='%.3f', key='Ac')
                As = st.number_input('Effective steel area As (mm²)', min_value=0.001, value=1967.0, format='%.3f', key='As')
                effective_areas = (Ac, As)
            else:
                effective_areas = None
            st.caption('Ac = π(D − 2t)²/4; As = π[D² − (D − 2t)²]/4.')
        with c3:
            st.markdown('**Eccentricity and corrugation**')
            if eccentricity_mode == 'e/r directly':
                e_ratio = st.number_input('e/r — eccentricity ratio', min_value=0.0, value=0.2, format='%.4f', key='e_ratio')
            else:
                e_mm = st.number_input('e — eccentricity (mm)', min_value=0.0, value=30.0, format='%.3f', key='e_mm')
                e_ratio = e_mm / (D / 2)
            if angle_mode == 'Angle in degrees':
                angle = st.number_input('Helix angle (degrees)', min_value=0.0, max_value=90.0, value=0.0, format='%.3f', key='angle')
            else:
                K = st.number_input('K — wave number', min_value=0.0, value=10.0, format='%.3f', key='wave_number')
                angle = math.degrees(math.atan(spacing * K / (math.pi * D)))
    else:
        columns = st.columns(3)
        labels = {'Helix Angle': 'Helix angle (degrees)', 'fcuAc': 'fcu × Ac (N)', 'fyAs': 'fy × As (N)', 'Concrete Types': 'Rubber replacement ratio', 'Confinement Factor': 'Confinement factor ξ'}
        features = {}
        for i, name in enumerate(STATIC):
            with columns[i // 3]:
                features[name] = st.number_input(labels.get(name, name), value=float(MODEL_INFO[scheme]['defaults'][name]), format='%.8g', key='feature_' + name)
    st.markdown('**Displayed strain window**')
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.number_input('Start strain', value=0.0, min_value=0.0, max_value=0.03, format='%.6f', key='start')
    with c2:
        end = st.number_input('End strain', value=0.03, min_value=0.0, max_value=0.03, format='%.6f', key='end')
    with c3:
        count = st.number_input('Displayed points', value=101, min_value=2, max_value=1000, step=1, key='points')
    submitted = st.form_submit_button('Predict curve', type='primary', use_container_width=True)
if submitted:
    clear_prediction()
    try:
        if mode == 'Geometry and materials':
            features, geometry = geometry_to_features(D, t, L, h, spacing, fcu, fy, rubber, e_ratio, angle, effective_areas)
        with st.spinner('Preparing the model and predicting the curve…'):
            result = run_prediction(scheme, features, start, end, count)
        result['geometry'] = geometry
        st.session_state['prediction'] = result
    except Exception as exc:
        st.error(str(exc))
if 'prediction' not in st.session_state:
    st.caption('Enter parameters and click Predict curve to display the result.')
    st.stop()
result = st.session_state['prediction']
st.subheader(f'Prediction results — V2.1 / Scheme {result['protocol']}')
outliers = range_checks(result['protocol'], result['features'])
if outliers:
    st.warning('Some inputs are outside this model’s development ranges.')
    st.dataframe(outliers, use_container_width=True, hide_index=True)
columns = st.columns(3)
columns[0].metric('Full-curve peak (0–0.03)', f'{result['capacity_kN']:.2f} kN')
columns[1].metric('Full-curve peak strain', f'{result['peak_strain']:.6f}')
columns[2].metric('Displayed-window maximum', f'{result['window_peak_kN']:.2f} kN')
st.caption('The complete curve uses a fixed 101-point grid on 0–0.03. Other windows are interpolation queries; they do not change the full-curve peak. Curves are displayed without smoothing.')
left, right = st.columns([2, 1])
with left:
    show_band = st.checkbox('Show prediction band', value=True)
    x = {'field': 'Strain', 'type': 'quantitative', 'axis': {'title': 'Strain'}}
    layers = []
    if show_band:
        layers.append({'mark': {'type': 'area', 'color': '#D8E4F0', 'opacity': 0.65}, 'encoding': {'x': x, 'y': {'field': 'Lower_kN', 'type': 'quantitative', 'axis': {'title': 'Load (kN)'}}, 'y2': {'field': 'Upper_kN'}}})
    layers.append({'mark': {'type': 'line', 'color': '#E07A5F', 'strokeWidth': 2}, 'encoding': {'x': x, 'y': {'field': 'Predicted_load_kN', 'type': 'quantitative', 'scale': {'zero': True}, 'axis': {'title': 'Load (kN)'}}}})
    st.vega_lite_chart(result['display'], {'layer': layers, 'height': 360}, use_container_width=True)
    st.download_button('Download curve image (SVG)', curve_svg(result, show_band), file_name=f'V2.1_{result['protocol']}_curve.svg', mime='image/svg+xml')
with right:
    st.dataframe([{key: row[key] for key in ['Strain', 'Predicted_load_kN', 'Lower_kN', 'Upper_kN']} for row in result['display']], use_container_width=True, hide_index=True, height=330)
    st.download_button('Download displayed curve CSV', to_csv(result['display']), file_name=f'V2.1_{result['protocol']}_displayed_curve.csv', mime='text/csv')
    st.download_button('Download full 101-point curve CSV', to_csv(result['full']), file_name=f'V2.1_{result['protocol']}_full_curve.csv', mime='text/csv')
with st.expander('Inputs used for this prediction'):
    st.dataframe([result['features']], use_container_width=True, hide_index=True)
    if result['geometry']:
        st.json(result['geometry'])
