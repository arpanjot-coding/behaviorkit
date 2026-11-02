"""Create the downloadable source archive, excluding weights and local state."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
target=ROOT.parent/'behaviorkit-prototype-source.zip'
excluded={'.git','__pycache__','.pytest_cache','models','media','profiles','build','dist','.openai','.venv','.sites-local'}
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(ROOT.rglob('*')):
        relative=path.relative_to(ROOT)
        if path.is_symlink() or not path.is_file() or any(part in excluded or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.suffix in {'.onnx','.pth','.pt','.engine','.pyc'}:
            continue
        archive.write(path,Path('behaviorkit-prototype')/relative)
print(target)
