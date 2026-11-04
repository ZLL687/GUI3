# -*- coding: utf-8 -*-
"""V2.2 Streamlit application with unchanged local frozen A/B model package.

The frontend uses Python 3.12–3.14; a dedicated Python 3.12 model environment
is prepared on first prediction. No model training or remote model URL.
"""
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
import csv,hashlib,io,math,os,sys,json,shutil,subprocess,tempfile,time,uuid
sys.dont_write_bytecode=True
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name,'1')


STATIC = ['D/t', 'L/D', 'Helix Angle', 'fcuAc', 'fyAs', 'h/l', 'Concrete Types', 'e/r', 'Confinement Factor']

MODEL_INFO = {'A': {'sha256': 'de80aeb0c6c0aaea8e5de0c75b911ecd87dc56270c668810e98f42602b40c018', 'q_joint': 1.1003871283195499, 'ranges': {'D/t': [8.3, 628.5], 'L/D': [1.19, 18.0], 'Helix Angle': [0.0, 41.4], 'fcuAc': [73941.57, 61819731.25], 'fyAs': [91060.0, 15087072.0], 'h/l': [0.0, 0.33], 'Concrete Types': [0.0, 0.3], 'e/r': [0.0, 1.25], 'Confinement Factor': [0.0621, 4.7021]}, 'defaults': {'D/t': 157.5, 'L/D': 2.38, 'Helix Angle': 0.0, 'fcuAc': 3076226.6, 'fyAs': 900634.9, 'h/l': 0.19, 'Concrete Types': 0.2, 'e/r': 0.0, 'Confinement Factor': 0.2928}, 'default_reference_specimen': 310}, 'B': {'sha256': '73267ef3c2d87988967fdb8fedc1d2527946d17eb5316742949173f8aefbdf2a', 'q_joint': 0.7367651980711041, 'ranges': {'D/t': [8.3, 300.0], 'L/D': [1.82, 18.0], 'Helix Angle': [0.0, 37.5], 'fcuAc': [73941.57, 18336842.94], 'fyAs': [91060.0, 15087072.0], 'h/l': [0.0, 0.2], 'Concrete Types': [0.0, 0.6], 'e/r': [0.0, 1.2], 'Confinement Factor': [0.0685, 4.7021]}, 'defaults': {'D/t': 37.75, 'L/D': 3.31, 'Helix Angle': 0.0, 'fcuAc': 1358741.51, 'fyAs': 830693.28, 'h/l': 0.0, 'Concrete Types': 0.0, 'e/r': 0.0, 'Confinement Factor': 0.6114}, 'default_reference_specimen': 136}}
MODEL_PACKAGE_SHA256 = 'a452f499671d21948da7557a850e8570ed3f2da41c775ce42f79d81270df9cb8'

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

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

RUNTIME_PACKAGES = {
    'numpy': '1.26.4',
    'pandas': '2.2.2',
    'scipy': '1.13.1',
    'scikit-learn': '1.5.1',
    'joblib': '1.4.2',
    'xgboost-cpu': '3.0.1',
    'lightgbm': '4.6.0',
    'catboost': '1.2.8',
    'scikit-optimize': '0.10.2',
}


def _runtime_cache():
    root = Path(os.environ.get('V22_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'v22_prediction_runtime')))
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _process_environment():
    environment = os.environ.copy()
    for name in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX'):
        environment.pop(name, None)
    environment['PYTHONNOUSERSITE'] = '1'
    environment['PYTHONIOENCODING'] = 'utf-8'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    return environment


def _creation_flags():
    return subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def _run_setup(command, environment):
    print('[V2.2] Preparing model runtime...', flush=True)
    completed = subprocess.run(command, env=environment, capture_output=True,
                               text=True, encoding='utf-8', errors='replace',
                               timeout=900, creationflags=_creation_flags())
    if completed.returncode:
        detail = (completed.stdout + '\n' + completed.stderr)[-4000:]
        print('[V2.2] Runtime setup failed:\n' + detail, file=sys.stderr, flush=True)
        raise RuntimeError('Model preparation could not finish. Check the application logs and retry.')
    return completed.stdout


@contextmanager
def _setup_lock(path):
    """Release the installation lock automatically if the process exits."""
    with Path(path).open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        started = time.monotonic()
        while True:
            try:
                handle.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() - started > 1800:
                    raise RuntimeError('Model preparation is busy. Retry after the current operation finishes.')
                time.sleep(0.25)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _runtime_ready(python, environment):
    if not Path(python).is_file():
        return False
    probe = (
        'import sys,importlib.metadata as m;'
        'assert sys.version_info[:2]==(3,12);'
        f'expected={RUNTIME_PACKAGES!r};'
        "versions={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()};"
        "assert all(versions.get(k,versions.get('xgboost') if k=='xgboost-cpu' else None)==v for k,v in expected.items())"
    )
    try:
        completed = subprocess.run([str(python), '-I', '-c', probe], env=environment,
                                   capture_output=True, timeout=30,
                                   creationflags=_creation_flags())
        return completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def prepare_model_runtime():
    """Keep fitted-model dependencies independent of the Streamlit interpreter."""
    root = _runtime_cache()
    environment = _process_environment()
    preferred = os.environ.get('V22_MODEL_PYTHON')
    if preferred:
        if not _runtime_ready(preferred, environment):
            raise RuntimeError('The specified model Python does not match the pinned V2.2 environment.')
        return str(Path(preferred).resolve())
    if _runtime_ready(sys.executable, environment):
        return sys.executable
    environment.update({
        'UV_CACHE_DIR': str(root / 'uv_cache'),
        'UV_PYTHON_INSTALL_DIR': str(root / 'python'),
        'UV_PYTHON_BIN_DIR': str(root / 'bin'),
        'UV_PYTHON_INSTALL_REGISTRY': '0',
        'UV_PYTHON_NO_REGISTRY': '1',
        'UV_PYTHON_DOWNLOADS': 'automatic',
    })
    recipe = json.dumps({'python': '3.12', 'packages': RUNTIME_PACKAGES}, sort_keys=True)
    fingerprint = hashlib.sha256(recipe.encode()).hexdigest()[:16]
    ready = root / ('ready_' + fingerprint + '.json')
    with _setup_lock(root / ('setup_' + fingerprint + '.lock')):
        if ready.is_file():
            try:
                saved = json.loads(ready.read_text(encoding='utf-8'))
                python = Path(saved['python'])
                if saved.get('recipe') == recipe and python.resolve().is_relative_to(root):
                    if _runtime_ready(python, environment):
                        return str(python)
            except (OSError, ValueError, KeyError):
                pass
        executable = shutil.which('uv')
        if executable:
            installer = [executable, '--no-config']
        else:
            import importlib.util
            if importlib.util.find_spec('uv') is None:
                raise RuntimeError('Upload the matching requirements.txt file and restart the application.')
            installer = [sys.executable, '-m', 'uv', '--no-config']
        folder = root / ('env_py312_' + fingerprint)
        python = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if folder.exists() and not python.is_file():
            folder = root / ('env_py312_' + fingerprint + '_' + uuid.uuid4().hex[:8])
            python = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not python.is_file():
            base = ['--python', sys.executable] if sys.version_info[:2] == (3, 12) else ['--managed-python', '--python', '3.12']
            _run_setup(installer + ['venv', *base, str(folder)], environment)
        packages = [name + '==' + version for name, version in RUNTIME_PACKAGES.items()]
        _run_setup(installer + ['pip', 'install', '--python', str(python),
                               '--index-url', 'https://pypi.org/simple',
                               '--only-binary=:all:', '--no-progress', *packages], environment)
        if not _runtime_ready(python, environment):
            raise RuntimeError('The model environment did not pass its startup check. See the application logs.')
        temporary = ready.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps({'recipe': recipe, 'python': str(python)}), encoding='utf-8')
        os.replace(temporary, ready)
    return str(python)




"""V2.2 frozen prediction backend; scientific imports occur only in the worker."""

def _prepare_assets(scheme, directory):
    package_path=Path(directory)/'V2.2_AB_models.joblib'
    if not package_path.is_file():raise FileNotFoundError('Place V2.2_AB_models.joblib beside streamlitapp.py.')
    if _sha(package_path)!=MODEL_PACKAGE_SHA256:raise ValueError('Model package mismatch. Upload the original V2.2 file unchanged.')
    target=_runtime_cache()/('five_file_assets_'+scheme+'_'+MODEL_PACKAGE_SHA256[:16]);ready=target/'ready.json'
    with _setup_lock(_runtime_cache()/('five_file_assets_'+scheme+'.lock')):
        target.mkdir(parents=True,exist_ok=True)
        if not ready.is_file():
            import joblib
            package=joblib.load(package_path)
            if package.get('format')!='V22_AB_FROZEN_MODEL_PACKAGE' or package.get('version')!='V2.2' or package.get('model_info')!=MODEL_INFO:raise ValueError('Invalid V2.2 model package.')
            for name,content in package['source_code'].items():
                path=target/'code'/name
                if not path.resolve().is_relative_to((target/'code').resolve()):raise ValueError('Invalid model code path.')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            native=package['native_models'][scheme]
            if hashlib.sha256(native).hexdigest()!=MODEL_INFO[scheme]['sha256']:raise ValueError('Frozen model content mismatch.')
            (target/'model.joblib').write_bytes(native)
            ready.write_text(json.dumps({'package_sha256':MODEL_PACKAGE_SHA256}),encoding='utf-8')
        else:
            if json.loads(ready.read_text(encoding='utf-8'))['package_sha256']!=MODEL_PACKAGE_SHA256:raise ValueError('Cached model package mismatch.')
        if _sha(target/'model.joblib')!=MODEL_INFO[scheme]['sha256']:raise ValueError('Frozen model checksum mismatch.')
    return target

class ModelBundle:
    """Lazy native A/B models; IDs group curves and never select saved predictions."""
    def __init__(self,directory=None):
        self.directory=Path(directory or Path(__file__).resolve().parent);self.models={};self._lock=RLock()
    def load(self,scheme):
        if scheme not in MODEL_INFO:raise ValueError('Select Scheme A or Scheme B.')
        if scheme not in self.models:
            assets=_prepare_assets(scheme,self.directory);sys.path.insert(0,str(assets/'code'))
            import runtime
            import joblib
            self.models[scheme]=joblib.load(assets/'model.joblib')
            if not math.isclose(float(self.models[scheme].q_joint_),MODEL_INFO[scheme]['q_joint'],rel_tol=1e-13):raise ValueError('Frozen interval parameters do not match this application.')
        return self.models[scheme]
    def predict(self,scheme,features,start=0.,end=.03,count=101,reference_grid=None):
        with self._lock:return predict_curve(self.load(scheme),scheme,features,start,end,count,reference_grid)

def predict_curve(model,scheme,features,start=0.,end=.03,count=101,reference_grid=None):
    import numpy as np
    import pandas as pd
    values=validate_features(features);validate_window(start,end,count)
    grid=np.linspace(0.,.03,101) if reference_grid is None else np.asarray(reference_grid,float)
    if grid.ndim!=1 or len(grid)<3 or not np.isfinite(grid).all() or grid[0]!=0 or abs(grid[-1]-.03)>1e-12 or not (np.diff(grid)>0).all():raise ValueError('The complete reference grid must increase from 0 to 0.03.')
    full=pd.DataFrame({key:np.repeat(v,len(grid)) for key,v in values.items()});full['Strain']=grid;full['Specimen ID']=0
    detail=model.curve_interval_details(full);prediction,lower,upper=[np.asarray(detail[k],float) for k in ['point','lower','upper']]
    capacity_detail=model.capacity_interval_details(full.iloc[[0]])
    capacity,capacity_lower,capacity_upper=[float(np.asarray(capacity_detail[k])[0]) for k in ['point','lower','upper']]
    if any(a.shape!=grid.shape or not np.isfinite(a).all() for a in [prediction,lower,upper]):raise ValueError('The model returned an invalid curve.')
    if prediction[0]!=0 or lower[0]!=0 or upper[0]!=0 or not np.isclose(prediction.max(),capacity,rtol=1e-10,atol=1e-7):raise ValueError('The full-curve and capacity outputs are inconsistent.')
    if (lower<0).any() or (lower>prediction+1e-8).any() or (upper<prediction-1e-8).any():raise ValueError('The frozen envelope failed its containment check.')
    q=float(model.q_joint_)
    def records(x):
        p,lo,hi=[np.interp(x,grid,y) for y in [prediction,lower,upper]]
        return [{'Strain':float(a),'Predicted_load_kN':float(b),'Lower_kN':float(c),'Upper_kN':float(d),'Model_version':'V2.2','Scheme':scheme,'Full_curve_peak_kN':capacity,'Capacity_lower90_kN':capacity_lower,'Capacity_upper90_kN':capacity_upper,'q_joint':q,'Interval_method':'frozen13_source_pilot_envelope_empirical','Reference_grid':'uniform101' if reference_grid is None else 'supplied_complete_grid',**values} for a,b,c,d in zip(x,p,lo,hi)]
    displayed=records(np.linspace(float(start),float(end),int(count)))
    return {'display':displayed,'full':records(grid),'features':values,'protocol':scheme,'capacity_kN':capacity,'capacity_lower90_kN':capacity_lower,'capacity_upper90_kN':capacity_upper,'peak_strain':float(grid[np.argmax(prediction)]),'window_peak_kN':max(row['Predicted_load_kN'] for row in displayed),'q_joint':q,'reference_grid':'uniform101' if reference_grid is None else 'supplied_complete_grid'}

class ModelService:
    """Keep the frontend independent of pinned Python 3.12 model dependencies."""
    def __init__(self,directory=None):
        self.directory=Path(directory or Path(__file__).resolve().parent).resolve()
        path=self.directory/'V2.2_AB_models.joblib'
        if not path.is_file():raise FileNotFoundError('Upload V2.2_AB_models.joblib beside streamlitapp.py.')
        if _sha(path)!=MODEL_PACKAGE_SHA256:raise ValueError('Model package mismatch. Upload the original V2.2 file unchanged.')
        self.python=prepare_model_runtime();self._lock=RLock()
    def predict(self,scheme,features,start=0.,end=.03,count=101,reference_grid=None):
        if scheme not in MODEL_INFO:raise ValueError('Select Scheme A or Scheme B.')
        values=validate_features(features);validate_window(start,end,count)
        request={'directory':str(self.directory),'scheme':scheme,'features':values,'start':float(start),'end':float(end),'count':int(count),'reference_grid':reference_grid}
        with self._lock:
            try:
                completed=subprocess.run([self.python,'-I','-u',str(Path(__file__).resolve()),'--worker'],input=json.dumps(request,allow_nan=False),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240,env=_process_environment(),creationflags=_creation_flags())
            except subprocess.TimeoutExpired as exc:raise RuntimeError('The model took too long to respond. Please retry.') from exc
        prefix='__V22_RESULT__';lines=[line[len(prefix):] for line in completed.stdout.splitlines() if line.startswith(prefix)]
        if completed.returncode or not lines:
            print('[V2.2] Prediction worker failed:\n'+completed.stderr[-4000:],file=sys.stderr,flush=True);raise RuntimeError('The prediction could not finish. See the application logs.')
        reply=json.loads(lines[-1])
        if not reply.get('ok'):raise RuntimeError(reply.get('error','Invalid model result.'))
        return reply['result']

def _worker_main():
    sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
    try:
        request=json.loads(sys.stdin.read());result=ModelBundle(request['directory']).predict(request['scheme'],request['features'],request['start'],request['end'],request['count'],request.get('reference_grid'));reply={'ok':True,'result':result}
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr);reply={'ok':False,'error':str(exc)}
    print('__V22_RESULT__'+json.dumps(reply,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__' and '--worker' in sys.argv:
    _worker_main();raise SystemExit(0)


# Streamlit interface

from html import escape
import streamlit as st
st.set_page_config(page_title='CFST Load–Strain Prediction | V2.2', page_icon='🏗️', layout='wide')

def clear_prediction():
    st.session_state.pop('prediction', None)

@st.cache_resource(show_spinner=False)
def get_model():
    return ModelService()

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
    parts.append(f'<text x="400" y="25" text-anchor="middle" font-size="17">{escape('V2.2 — Scheme ' + result['protocol'])}</text>')
    parts.append('<text x="420" y="465" text-anchor="middle">Strain</text>')
    parts.append('<text x="20" y="230" transform="rotate(-90 20 230)" text-anchor="middle">Load (kN)</text>')
    parts.append('</g></svg>')
    return ''.join(parts).encode('utf-8')
st.title('CFST Load–Strain Prediction')
st.caption('MultiPhysics-DGS-CA · V2.2')
st.caption('Frozen model · Original 815-specimen database · Experimental and original FE records')
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
    st.caption('The first prediction may take longer while the model service is prepared.')
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
    submitted = st.form_submit_button('Predict curve', type='primary', width='stretch')
if submitted:
    clear_prediction()
    try:
        if mode == 'Geometry and materials':
            features, geometry = geometry_to_features(D, t, L, h, spacing, fcu, fy, rubber, e_ratio, angle, effective_areas)
        with st.spinner('Preparing the model and predicting the curve…', show_time=True):
            result = run_prediction(scheme, features, start, end, count)
        result['geometry'] = geometry
        st.session_state['prediction'] = result
    except Exception as exc:
        st.error(str(exc))
if 'prediction' not in st.session_state:
    st.caption('Enter parameters and click Predict curve to display the result.')
    st.stop()
result = st.session_state['prediction']
st.subheader(f'Prediction results — V2.2 / Scheme {result['protocol']}')
outliers = range_checks(result['protocol'], result['features'])
if outliers:
    st.warning('Some inputs are outside this model’s development ranges.')
    st.dataframe(outliers, width='stretch', hide_index=True)
columns = st.columns(3)
columns[0].metric('Full-curve peak (0–0.03)', f'{result['capacity_kN']:.2f} kN')
columns[1].metric('Full-curve peak strain', f'{result['peak_strain']:.6f}')
columns[2].metric('Displayed-window maximum', f'{result['window_peak_kN']:.2f} kN')
st.caption('New-parameter predictions use a uniform 101-point reference grid on 0–0.03. Display windows interpolate this complete curve and preserve its peak; no smoothing is applied.')
st.caption(f"Capacity envelope: {result['capacity_lower90_kN']:.2f}–{result['capacity_upper90_kN']:.2f} kN. The 90% level is empirical; independent split-conformal coverage is not claimed.")
left, right = st.columns([2, 1])
with left:
    show_band = st.checkbox('Show empirical 90% envelope', value=True)
    x = {'field': 'Strain', 'type': 'quantitative', 'axis': {'title': 'Strain'}}
    layers = []
    if show_band:
        layers.append({'mark': {'type': 'area', 'color': '#D8E4F0', 'opacity': 0.65}, 'encoding': {'x': x, 'y': {'field': 'Lower_kN', 'type': 'quantitative', 'axis': {'title': 'Load (kN)'}}, 'y2': {'field': 'Upper_kN'}}})
    layers.append({'mark': {'type': 'line', 'color': '#E07A5F', 'strokeWidth': 2}, 'encoding': {'x': x, 'y': {'field': 'Predicted_load_kN', 'type': 'quantitative', 'scale': {'zero': True}, 'axis': {'title': 'Load (kN)'}}}})
    st.vega_lite_chart(result['display'], {'layer': layers, 'height': 360}, width='stretch')
    st.download_button('Download curve image (SVG)', curve_svg(result, show_band), file_name=f'V2.2_{result['protocol']}_curve.svg', mime='image/svg+xml')
with right:
    st.dataframe([{key: row[key] for key in ['Strain', 'Predicted_load_kN', 'Lower_kN', 'Upper_kN']} for row in result['display']], width='stretch', hide_index=True, height=330)
    st.download_button('Download displayed curve CSV', to_csv(result['display']), file_name=f'V2.2_{result['protocol']}_displayed_curve.csv', mime='text/csv')
    st.download_button('Download full 101-point curve CSV', to_csv(result['full']), file_name=f'V2.2_{result['protocol']}_full_curve.csv', mime='text/csv')
with st.expander('Inputs used for this prediction'):
    st.dataframe([result['features']], width='stretch', hide_index=True)
    if result['geometry']:
        st.json(result['geometry'])
