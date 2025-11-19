import importlib, traceback
pkgs = ['pandas','sklearn','streamlit','requests','numpy','matplotlib']
for p in pkgs:
    try:
        m = importlib.import_module(p)
        print(p, getattr(m, '__version__', None))
    except Exception as e:
        print('FAILED', p)
        traceback.print_exc()
