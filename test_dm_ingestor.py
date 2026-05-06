import sys
from pathlib import Path
from PIL import Image

# Local paths needed to import the ingestor class and its dependencies
repo_path = Path(__file__).parent.parent
sys.path.append(str(repo_path))

from ingestors import dm_ingestor

data_path = Path('/home/percius/scripting/openNCEM/ncempy/data')
file_paths = (Path('dmTest_3D_int16_64,65,66.dm3'),
              Path('08_carbon.dm3'))

for file_path in file_paths:
    ingestor = dm_ingestor.DigitalMicrographIngestor(file_to_upload=str(data_path / file_path))
    assert ingestor.is_file_supported()

    ingestor.get_scientific_metadata()
    assert 'Calibrations Brightness Origin' in ingestor.scientific_metadata

    ingestor.get_dataset_metadata()
    print(ingestor.dataset_name)
    assert ingestor.dataset_name == file_path.name

    ingestor.get_data_files()

    im = ingestor.generate_dm_thumbnail()
    assert isinstance(im, Image.Image)