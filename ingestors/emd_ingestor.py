import io

from pathlib import Path
from PIL import Image
import ncempy.io as nio
import matplotlib.pyplot as plt
import logging

from ingestors.crucible_ingestor import CrucibleDatasetIngestor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BerkeleyEmdIngestor(CrucibleDatasetIngestor):
    '''subclass for ingesting Berkeley EMD files'''

    def is_file_supported(self):
        if not self.file_to_upload.endswith('.emd'):
            return False
        
        with nio.emd.fileEMD(self.file_to_upload, readonly=True) as emd1:
            if len(emd1.list_emds) > 0:
                return True
            else:
                return False

    
    def get_scientific_metadata(self):
        with nio.emd.fileEMD(self.file_to_upload, readonly=True) as emd1:
            self.scientific_metadata.update(emd1.getMetadata(0))
        logger.info(f'Got metadata from Berkeley EMD: {self.scientific_metadata=}')


    def get_dataset_metadata(self):
         # Use parent class method to set data_format, size, and source_folder
        CrucibleDatasetIngestor.get_dataset_metadata(self)
        self.dataset_name = Path(self.file_to_upload).name
        # TODO: parse this
        self.measurement = ''


    def generate_thumbnail(self, target_size=(200, 200), dpi=100):
        """Generate a thumbnail from an EMD image as a PNG.

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
            with nio.emd.fileEMD(self.file_to_upload, readonly=True) as emd1:
                data_array, _ = emd1.get_emdgroup(0)
            
            if data_array is None:
                raise ValueError("No data found in EMD file.")
            
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
            fg.savefig(buf, bbox_inches='tight', pad_inches=0.05, dpi=dpi)
            im = Image.open(buf)
            return im
        except Exception as e:
            print(f"Failed to generate thumbnail: {e}")

    def get_thumbnails(self):
        try:
            thumbnail = self.generate_thumbnail()
            if thumbnail:
                self.add_thumbnail(thumbnail, "EMD_Thumbnail")
        except Exception as e:
            print(f"Failed to extract thumbnail: {e}")