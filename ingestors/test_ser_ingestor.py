import sys
from pathlib import Path
from PIL import Image

# Local paths needed to import the SerIngestor class and its dependencies
repo_path = Path(__file__).parent.parent.parent
sys.path.append(str(repo_path / 'ingestion-consumer'))
sys.path.append(str(repo_path))

import ser_ingestor

data_path = Path('/home/percius/scripting/openNCEM/ncempy/data')
file_paths = (Path('01_Si110_5images_1.ser'),
              Path('16_STOimage_1.ser'))

for file_path in file_paths:
    ingestor = ser_ingestor.SerIngestor(str(data_path / file_path))

    assert ingestor.is_file_supported()

    ingestor.get_scientific_metadata()
    assert 'AcceleratingVoltage' in ingestor.scientific_metadata

    ingestor.get_dataset_metadata()
    assert ingestor.dataset_name == file_path.stem

    ingestor.get_data_files()

    im = ingestor.generate_thumbnail()
    assert isinstance(im, Image.Image)