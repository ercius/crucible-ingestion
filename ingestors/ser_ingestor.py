import io
from datetime import datetime as dt
from pathlib import Path
from typing import ClassVar

from PIL import Image
import logging
import numpy as np
import ncempy.io as nio
import matplotlib.pyplot as plt

from ingestors.crucible_ingestor import CrucibleDatasetIngestor


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SerIngestor(CrucibleDatasetIngestor):
    '''subclass for ingesting ser / emi files'''
    
    supported_filetypes: ClassVar[list[str]] = ['ser']
    
    def is_file_supported(self):
        return np.any([self.file_to_upload.endswith(ftype)
                       for ftype in self.supported_filetypes])


    def get_scientific_metadata(self):
        """Extract scientific metadata from the ser file using ncempy."""
        with nio.ser.fileSER(self.file_to_upload) as ser:
            self.scientific_metadata.update(ser.getMetadata())

    
    def get_dataset_metadata(self):
        '''
        Set the structured metadata according to Crucible's schema.
        Suggested ones are: dataset_name, instrument_name, measurement, 
        session_name, timestamp, data_format, source_folder
        '''
        # Use parent class method to set data_format, size, and source_folder
        CrucibleDatasetIngestor.get_dataset_metadata(self)
        
        acquired_date = self.scientific_metadata.get('AcquireDate')
        if acquired_date:
            tia_date_format = "%a %b %d %H:%M:%S %Y"
            self.timestamp = dt.strptime(acquired_date, tia_date_format).isoformat()

        self.measurement = self.scientific_metadata.get('Mode []')
        self.dataset_name = Path(self.file_to_upload).name

    def generate_thumbnail(self, target_size=(200, 200), dpi=100):
        
        fig_size = (target_size[0] / dpi, target_size[1] / dpi) # inches
        with nio.ser.fileSER(self.file_to_upload) as ser:
            data_array = ser.getDataset(0)[0]

        # Use the middle slice if it's a 3D array, otherwise use the 2D array directly
        if data_array.ndim > 2:
            # Calculate the middle index for all dimensions EXCEPT the last two
            middle_indices = tuple(dim // 2 for dim in data_array.shape[:-2])
            # Unpack the tuple to slice the array.
            image_array = data_array[middle_indices]
        else:
            image_array = data_array

        fg, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
        # Plot an image if it's 2D, otherwise plot a line graph for 1D data
        if image_array.ndim == 2:
            ax.imshow(image_array, cmap='gray')
            ax.axis('off')
        elif image_array.ndim == 1:
            ax.plot(image_array)
        else:
            raise ValueError("Data array has unsupported number of dimensions for thumbnail generation.")

        # Convert to PIL Image and store in self.thumbnails
        buf = io.BytesIO()
        fg.savefig(buf, bbox_inches='tight', pad_inches=0.05, dpi=100)
        im = Image.open(buf)
        return im


    def get_thumbnails(self):
        try:
            thumbnail = self.generate_thumbnail()
            if thumbnail:
                self.add_thumbnail(thumbnail, "TIA_Thumbnail")
        except Exception as e:
            print(f"failed to extract thumbnail: {e}")





    

    
    
    





