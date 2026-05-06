
import io
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from PIL import Image
import numpy as np
import logging
import ncempy.io as nio
import matplotlib.pyplot as plt

from ingestors.crucible_ingestor import CrucibleDatasetIngestor


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DigitalMicrographIngestor(CrucibleDatasetIngestor):
    ''' Ingestor for Digital Micrograph files (DM3 and DM4) using ncempy.
    Supported file types: DM3, DM4
    '''

    supported_filetypes: ClassVar[list[str]] = ['dm3', 'dm4']

    def is_file_supported(self):
        """Check if the file to upload is a supported DM file based on its extension."""
        return np.any([self.file_to_upload.endswith(ftype) for ftype in self.supported_filetypes])

   
    def get_scientific_metadata(self):
        """Extract scientific metadata from the DM file using ncempy."""
        meta_data = {}
        with nio.dm.fileDM(self.file_to_upload, on_memory=True) as dm1:
            md = dm1.getMetadata(0)
            meta_data.update(md)
        self.scientific_metadata = meta_data


    def get_dataset_metadata(self):
        '''
        Set the structured metadata according to Crucible's schema.
        Suggested ones are: dataset_name, instrument_name, measurement, 
        session_name, timestamp, data_format, source_folder
        '''
        # Use parent class method to set data_format, size, and source_folder
        CrucibleDatasetIngestor.get_dataset_metadata(self)
        
        # Create a datetime from the acquisition date and time if both are available in the metadata
        if 'DataBar Acquisition Date' in self.scientific_metadata and \
           'DataBar Acquisition Time' in self.scientific_metadata:
            dt = datetime.strptime(
            f"{self.scientific_metadata['DataBar Acquisition Date'].strip()} {self.scientific_metadata['DataBar Acquisition Time'].strip()}",
            "%m/%d/%Y %I:%M:%S %p")
            self.timestamp = dt.isoformat()

        # Define the "measurement" based on the TEM lens setup
        illumination_mode = self.scientific_metadata.get('Microscope Info Illumination Mode', '')
        imaging_mode = self.scientific_metadata.get('Microscope Info Imaging Mode', '')
        self.measurement = f"{illumination_mode} {imaging_mode}".strip()
        logger.info(f'{self.measurement=}')
        self.dataset_name = Path(self.file_to_upload).name # file name without extension


    def generate_dm_thumbnail(self, target_size=(200, 200), dpi=100):
        """Generate a thumbnail from a DM image as a PNG.
        
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
        try:
            with nio.dm.fileDM(self.file_to_upload, on_memory=True) as f0:
                dd = f0.getSlice(0, 0) # ensures we get a 2D slice.
                image_array = dd['data']
            fg, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
            ax.imshow(image_array)
            ax.axis('off')

            # Convert to PIL Image and store in self.thumbnails
            buf = io.BytesIO()
            fg.savefig(buf, bbox_inches='tight', pad_inches=0.05, dpi=100)
            im = Image.open(buf)
            return im
        except Exception as e:
            print(f"Failed to generate thumbnail: {e}")

    def get_thumbnails(self):
        try:
            thumbnail = self.generate_dm_thumbnail()
            if thumbnail:
                self.add_thumbnail(thumbnail, "DM_Thumbnail")
        except Exception as e:
            print(f"Failed to extract thumbnail: {e}")

