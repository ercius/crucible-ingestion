import io

from pathlib import Path
from PIL import Image
import ncempy.io as nio
import matplotlib.pyplot as plt
import logging
import numpy as np

from .crucible_ingestor import CrucibleDatasetIngestor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MrcIngestor(CrucibleDatasetIngestor):
    '''subclass for ingesting MRC files'''

    def is_file_supported(self):
        if not Path(self.file_to_upload).suffix == '.mrc':
            return False
        try:
            with nio.mrc.fileMRC(self.file_to_upload) as mrc1:
                return True
        except Exception as e:
            logger.error(f"Error occurred while checking MRC file support: {e}")
            return False

    
    def get_scientific_metadata(self):
        with nio.mrc.fileMRC(self.file_to_upload) as mrc1:
            md = mrc1.getMetadata()
            for key, value in md.items():
                if isinstance(value, np.ndarray) and not value.flags.c_contiguous:
                    md[key] = list(value)
            self.scientific_metadata.update(md)
        logger.info(f'Got metadata from MRC: {self.scientific_metadata=}')

        file_path = Path(self.file_to_upload)
        # Read tilt angles from .rawtlt file if it exists
        rawtltName = file_path.with_suffix('.rawtlt')
        if rawtltName.exists():
            with open(rawtltName, 'r') as f1:
                tilts = list(map(float, f1))
            self.scientific_metadata['tilt angles'] = tilts
        
        # Read FEI parameters from .txt file if it exists
        FEIparameters = file_path.with_suffix('.txt')
        if FEIparameters.exists():
            try:
                with open(FEIparameters, 'r', encoding='utf-8-sig') as f2:
                    lines = f2.readlines()
            except UnicodeDecodeError:
                with open(FEIparameters, 'r', encoding='cp1252') as f2:
                    lines = f2.readlines()
            pp1 = list([ii[18:].strip().split(':')] for ii in lines[3:-1])
            pp2 = {}
            for ll in pp1:
                try:
                    pp2[ll[0]] = float(ll[1])
                except:
                    pass  # skip lines with no data
            self.scientific_metadata.update(pp2)

    def parse_measurement(self):
        # Test for metadata that is indicative of a tilt series from FEI tomo software.
        if self.scientific_metadata.get('axisOrientations') is not None and self.scientific_metadata.get('axisOrientations') is not None:
            self.measurement = 'tomography'
        else:
            self.measurement = None
        logger.info(f'{self.measurement=}')

    def get_dataset_metadata(self):
         # Use parent class method to set data_format and size
        CrucibleDatasetIngestor.get_dataset_metadata(self)
        self.dataset_name = Path(self.file_to_upload).name


    def generate_thumbnail(self, target_size=(200, 200), dpi=100):
        """Generate a thumbnail from an MRC image as a PNG.

        Parameters
        ----------
        target_size : tuple
            Desired size of the thumbnail in pixels (width, height).
        dpi : int
            Dots per inch for the thumbnail image.

        Returns
        -------
        : PIL.Image
            Thumbnail image as a PIL Image object.

        """

        fig_size = (target_size[0] / dpi, target_size[1] / dpi) # inches
       
        fg = None
        buf = None
        try:
            with nio.mrc.fileMRC(self.file_to_upload) as mrc1:
                image_array = mrc1.getSlice(mrc1.dataSize[0] // 2)  # Get the middle slice for 3D data, or the only slice for 2D data
            
            if image_array is None:
                raise ValueError("No data found in MRC file.")

            fg, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
            ax.imshow(image_array, cmap='gray')
            ax.axis('off')
            fg.tight_layout(pad=0.05)

            # Decode the rendered image, then enforce the requested resolution.
            buf = io.BytesIO()
            fg.savefig(buf, bbox_inches='tight', pad_inches=0.05, dpi=dpi)
            buf.seek(0)
            with Image.open(buf) as rendered_image:
                im = rendered_image.resize(target_size, Image.Resampling.LANCZOS)
            return im
        except Exception as e:
            logger.exception("Failed to generate thumbnail: %s", e)
            return None
        finally:
            if buf is not None:
                buf.close()
            if fg is not None:
                plt.close(fg)

    def get_thumbnails(self):
        try:
            thumbnail = self.generate_thumbnail()
            if thumbnail:
                self.add_thumbnail(thumbnail, "MRC_Thumbnail")
        except Exception as e:
            logger.exception("Failed to extract thumbnail: %s", e)