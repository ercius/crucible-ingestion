import sys
from pathlib import Path
from PIL import Image

# Local paths needed to import the ingestor class and its dependencies
repo_path = Path(__file__).parent.parent
sys.path.append(str(repo_path))

from ingestors import emd_ingestor

data_path = Path('/home/percius/scripting/openNCEM/ncempy/data')
file_paths = (Path('Acquisition_18.emd'),
              )

for file_path in file_paths:
    ingestor = emd_ingestor.BerkeleyEmdIngestor(file_to_upload=str(data_path / file_path))

    assert ingestor.is_file_supported()

    ingestor.get_scientific_metadata()
    assert 'magnification' in ingestor.scientific_metadata

    ingestor.get_dataset_metadata()
    assert ingestor.dataset_name == file_path.name

    ingestor.get_data_files()

    im = ingestor.generate_thumbnail()
    assert isinstance(im, Image.Image)