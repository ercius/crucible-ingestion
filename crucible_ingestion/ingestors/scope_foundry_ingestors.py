import os
import re
import functools
import logging
import requests
from io import BytesIO
from typing import ClassVar

import h5py
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
from .h5_ingestor import H5Ingestor
from .crucible_ingestor import TMP_DIR
from ..client import get_client
from crucible.models import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _is_mfid(val) -> bool:
    """True if val is an MFID: a 26-character Crockford Base32 encoding of a UUID.

    The h5 attributes are named *_uuid but hold MFIDs, which are not parseable
    by uuid.UUID().
    """
    return isinstance(val, str) and len(val) == 26 and val.isalnum()


def check_orcid_entry(orcid_string):
    """Validate and return an ORCID string if it matches the expected format."""
    if not isinstance(orcid_string, str):
        return None
    orcid_string = orcid_string.strip()
    if re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$', orcid_string):
        return orcid_string
    return None

class ScopeFoundryH5Ingestor(H5Ingestor):
    supported_measurements: ClassVar[list[str]] = ['simple_tiled_image', 
                                                    'canon_camera_capture', 
                                                    'picam_readout',
                                                    'm4_hyperspectral_2d_scan',
                                                    'andor_hyperspec_scan',
                                                    'hyperspectral_2d_scan',
                                                    'fiber_winspec_scan',
                                                    'hyperspec_picam_mcl',
                                                    'hyperspec_picam_mcl_sweep',
                                                    'asi_hyperspec_scan',
                                                    'asi_OO_hyperspec_scan',
                                                    'oo_asi_hyperspec_scan',
                                                    'andor_asi_hyperspec_scan', 
                                                    'ald_run_upd',
                                                    'ald_run',
                                                    'ald_run_measure'
                                                    ]
    
    def is_file_supported(self):
        if self.file_to_upload.endswith('h5'):
            return np.any([self.file_to_upload.endswith(f"{meas_name}.h5")
                           for meas_name in self.supported_measurements])
        return False
    
    def parse_measurement(self):
        self.measurement = self.h5file.visit(self._find_measurement)


    def get_dataset_metadata(self):
        self.instrument_name = self.scientific_metadata['app']['name']

        H5Ingestor.get_dataset_metadata(self)

        # overwrite unique ID if one is in the file
        if 'unique_id' in self.h5file.attrs.keys():
            self.unique_id = self.h5file.attrs['unique_id']

        # overwrite creation time and data format
        self.timestamp = datetime.fromtimestamp(self.h5file.attrs['time_id']).isoformat()
        self.data_format = "ScopeFoundryH5"

        # parse session_name and tags
        default_tags_value = "list,tags,separated,by,commas (optional)"
        default_session_value = "(optional)"

        try: 
            scope_foundry_tags = self.scientific_metadata['hardware']['mf-crucible']['settings']['tags'].strip()
            scope_foundry_session = self.scientific_metadata['hardware']['mf-crucible']['settings']['session_name'].strip()

        except Exception:
            logger.warning("no mf-crucible settings found for tags or session_name")
            scope_foundry_tags = default_tags_value
            scope_foundry_session = default_session_value

        if scope_foundry_tags != default_tags_value:
            self.keywords += [x.strip() for x in scope_foundry_tags.split(",")]

        if scope_foundry_session != default_session_value:
            self.session_name = scope_foundry_session
            self.keywords += [self.session_name]


    def _find_measurement(self,k): 
        """regular expression tree walking function that
        Finds the first measurement in a ScopeFoundry HDF5
        k: key / path of hdf5 object (dataset or group)
        """
        r = re.compile("measurement/[^/]*$")
        if re.match(r, k): 
            return(k.split("/")[1])   

    def parse_orcid(self):
        if self.owner_orcid:
            return
        self.owner_orcid = check_orcid_entry(self.scientific_metadata['hardware']['mf_crucible']['settings']['orcid'])
        return


    def parse_project_id(self):
        if self.project_id:
            return
        else:
             self.project_id = self.scientific_metadata['hardware']['mf_crucible']['settings']['proposal'].split(" ")[0]
        return 




class ALDScopeFoundryH5Ingestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['ald_run_upd', 'ald_run', 'ald_run_measure']
    creation_location: ClassVar[str] = "67-4210"


class SimpleTiledImageScopeFoundryH5Ingestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['simple_tiled_image']
    creation_location: ClassVar[str] = "67-1207"

    def get_thumbnails(self):
        tile_px = 64
        TILED_DSET = "measurement/simple_tiled_image/live_img_map"
        with h5py.File(self.file_to_upload, 'r') as h5file:

            if TILED_DSET not in h5file:
                return None
            imgs = h5file[TILED_DSET]
            shape = imgs.shape
            if len(shape) == 6:
                _, Nv, Nh, th, tw, _nc = shape
                rgb = (_nc == 3)
            elif len(shape) == 5:
                _, Nv, Nh, th, tw = shape
                rgb = False
            else:
                return None

            # Stride each tile down to ~tile_px on its short edge.
            step = max(1, min(th, tw) // tile_px)
            cell_h = len(range(0, th, step))
            cell_w = len(range(0, tw, step))

            if rgb:
                montage = np.zeros((Nv * cell_h, Nh * cell_w, 3), dtype=np.uint8)
            else:
                montage = np.zeros((Nv * cell_h, Nh * cell_w), dtype=np.uint8)

            for r in range(Nv):
                for c in range(Nh):
                    if rgb:
                        tile = np.asarray(imgs[0, r, c, ::step, ::step, :])
                    else:
                        tile = np.asarray(imgs[0, r, c, ::step, ::step])

                    if tile.dtype != np.uint8:
                        tmax = float(tile.max()) or 1.0
                        tile = (tile.astype(np.float32) / tmax * 255).astype(np.uint8)
                        
                    mr = Nv - 1 - r
                    h = min(cell_h, tile.shape[0])
                    w = min(cell_w, tile.shape[1])
                    montage[mr * cell_h:mr * cell_h + h,
                            c * cell_w:c * cell_w + w] = tile[:h, :w]

        self.add_thumbnail(Image.fromarray(montage), "Preview Image")


class CanonCaptureScopeFoundryH5Ingestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['canon_camera_capture']
    creation_location: ClassVar[str] = "67-1207"

    def get_thumbnails(self):
        image_file_name = f"{self.file_to_upload}.JPG"
        single_image = Image.open(image_file_name)
        self.add_thumbnail(single_image, "Canon Camera Capture")
        

class SingleSpecScopeFoundryH5Ingestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['picam_readout']
    creation_location: ClassVar[str] = "67-1217"

    def get_thumbnails(self):
        with h5py.File(self.file_to_upload, 'r') as h5file:
            M = h5file[f"measurement/{self.measurement}"]
            spec = np.array(M['spectrum'])
            raman = np.array(M['raman_shifts'])
        plt.plot(raman, spec)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        self.add_thumbnail(Image.open(buf), "Picam Readout")
        


class HyperspecScopeFoundryH5Ingestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['m4_hyperspectral_2d_scan',
                              'andor_hyperspec_scan',
                              'hyperspectral_2d_scan',
                              'fiber_winspec_scan',
                              'hyperspec_picam_mcl',
                              'asi_OO_hyperspec_scan',
                              'oo_asi_hyperspec_scan',
                              'andor_asi_hyperspec_scan']

    creation_location: ClassVar[str] = "67-1217"
    
    def get_thumbnails(self):
        try:
            with h5py.File(self.file_to_upload, 'r') as h5file:
                M = h5file[f'measurement/{self.measurement}']
                spec_map = np.array(M['spec_map'])[0]
                wls = np.array(M['wls'])

            buf = BytesIO()
            plt.imsave(buf, spec_map.sum(axis=-1), origin='lower', format='png')
            buf.seek(0)
            self.add_thumbnail(Image.open(buf), "Spectral Map")

            plt.plot(wls, spec_map.sum(axis=(0,1)))
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            self.add_thumbnail(Image.open(buf), "Sum of Spectra")
        except Exception as err:
            logger.error(f"failed to generate thumbnail for {self.file_to_upload} due to error {err}")



class HyperspecSweepScopeFoundryH5Ingestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['hyperspec_picam_mcl_sweep']
    creation_location: ClassVar[str] = "67-1217"
    
    def get_thumbnails(self):
        pass


class ToupcamLiveScopeFoundryH5Ingestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['toupcam_live']
    creation_location: ClassVar[str] = "67-1217"
    
    def get_thumbnails(self):
        with h5py.File(self.file_to_upload, 'r') as h5file:

            if 'image' in h5file['measurement']['toupcam_live'].keys():
                imarray = np.array(h5file['measurement']['toupcam_live']['image'])
            else:
                logger.info(f"{h5file['measurement']['toupcam_live'].keys()=}")
                return
            
        h5image = Image.fromarray(imarray)
        self.add_thumbnail(h5image, "Toupcam Live Image")


class CLSyncRasterScanIngestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['sync_raster_scan']
    creation_location: ClassVar[str] = '67-1210'
    
    def get_thumbnails(self):
        with h5py.File(self.file_to_upload, 'r') as h5file:
            M = h5file[f'measurement/{self.measurement}']
            if 'adc_map' in M.keys():
                adc_map = np.array(M['adc_map'])[0,0]
            else:
                adc_map = np.array([])
            
            if 'ctr_map' in M.keys():
                ctr_map = np.array(M['ctr_map'])[0,0]
            else:
                ctr_map = np.array([])
            logger.info(f"{adc_map.shape=}, {ctr_map.shape=}")

        # make a thumbnail for each channel in the ADC map
        for i in range(adc_map.shape[-1]):
            buf = BytesIO()
            plt.imsave(buf, adc_map[:,:,i], origin='lower', format='png')
            buf.seek(0)
            self.add_thumbnail(Image.open(buf), f"ADC Channel {i}")

        # make a thumbnail for each channel in the Counter map
        for i in range(ctr_map.shape[-1]):
            buf = BytesIO()
            plt.imsave(buf, ctr_map[:,:,i], origin='lower', format='png')
            buf.seek(0)
            self.add_thumbnail(Image.open(buf), f"Counter Channel {i}")


class CLHyperspecIngestor(ScopeFoundryH5Ingestor):

    supported_measurements: ClassVar[list[str]] = ['hyperspec_cl']
    creation_location: ClassVar[str] = '67-1210'

    def get_thumbnails(self):
        # Hyperspectral dataset include analog and counter
        # channels from sync_raster_scan, so we create thumbnails for those channels
        CLSyncRasterScanIngestor.get_thumbnails(self)

        with h5py.File(self.file_to_upload, 'r') as h5file:
            M = h5file[f'measurement/{self.measurement}']
            if not 'spec_map' in list(M.keys()):   
                return None
            spec_map = np.array(M['spec_map'])[0,0]
            wls = np.array(M['wls'])

        buf = BytesIO()
        plt.imsave(buf, spec_map.sum(axis=-1), origin='lower', format='png')
        buf.seek(0)
        self.add_thumbnail(Image.open(buf), "Spectral Map")

        plt.plot(wls, spec_map.sum(axis=(0,1)))
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        self.add_thumbnail(Image.open(buf), "Sum of Spectra")


class SpinBotIngestor(ScopeFoundryH5Ingestor):

    creation_location: ClassVar[str] = '67-4203'

    def get_dataset_metadata(self):
        ScopeFoundryH5Ingestor.get_dataset_metadata(self)
        
        default_tags_value = "list,tags,separated,by,commas (optional)"
        default_session_value = "(optional)"
        
        try: 
            scope_foundry_tags = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['tags'].strip()
            scope_foundry_session = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['session_name'].strip()

        except Exception:
            logger.warning("no mf-crucible settings found for tags or session_name")
            scope_foundry_tags = default_tags_value
            scope_foundry_session = default_session_value

        if scope_foundry_tags != default_tags_value:
            self.keywords += [x.strip() for x in scope_foundry_tags.split(",")]

        if scope_foundry_session != default_session_value:
            self.session_name = scope_foundry_session
            self.keywords += [self.session_name]

        self.keywords += [x for x in self.file_to_upload.split('/') if 'campaign' in x.lower()]
        self.keywords += [x for x in self.file_to_upload.split('/') if 'batch' in x.lower()]

    
    def parse_orcid(self):
        if self.owner_orcid:
            return
        self.owner_orcid = check_orcid_entry(self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['orcid'])
        return


    def parse_project_id(self):
        if self.project_id:
            return
        else:
            self.project_id = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['proposal'].split(" ")[0]
        return 

    def parse_batch(self):
        full_batch_id = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['batch_id']
        crucible_batch_id = full_batch_id.split('_')[2]
        batch_name = full_batch_id.split('_')[1]
        if not _is_mfid(crucible_batch_id):
            raise ValueError(f"batch id {crucible_batch_id!r} parsed from {full_batch_id!r} is not an MFID")
        owner_orcid = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['orcid']
        sample_info = {"unique_id": crucible_batch_id, "sample_name": batch_name, "owner_orcid": owner_orcid,
                       "project_id": self.project_id, "description": full_batch_id}
        self.batch = sample_info
        return(sample_info)

    def parse_samples(self):
        sample_label = self.scientific_metadata['app']['settings']['sample']
        logger.info(f"{sample_label=}")
        if sample_label is None:
            return
        elif len(sample_label) == 0:
            return
        else:
            logger.info(f"{sample_label}")
            
        owner_orcid = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['orcid']
        # the label must carry an MFID: without it samples cannot be deduplicated on re-ingest
        if not _is_mfid(sample_label):
            raise ValueError(f"sample label {sample_label!r} is not an MFID; cannot identify the sample")
        sample_id = sample_label

        sample = {"unique_id": sample_id,
                  "sample_name": sample_label, 
                  "owner_orcid": owner_orcid,
                  "project_id": self.project_id,
                  "parents": [{'unique_id': self.batch['unique_id']}]}
        
        # get the rest of the metadata
        self.samples.append(sample)
        return



class SpinbotSpecLineIngestor(SpinBotIngestor):

    supported_measurements: ClassVar[list[str]] = ['spec_line_scan']
 
    def get_thumbnails(self):
        try:
            with h5py.File(self.file_to_upload, 'r') as h5file:
                M = h5file[f"measurement/{self.measurement}"]
                spectra = np.array(M['spectra'])
                wls = np.array(M['wls'])
            for i in range(0, spectra.shape[0]):
                plt.plot(wls, spectra[i], label=f" spectra {i+1}")
            plt.legend()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            self.add_thumbnail(Image.open(buf), "SpinBot Spectra")
        except Exception as err:
            logger.error(f"failed to generate thumbnail for {self.file_to_upload} due to error {err}")



class SpinbotSpecRunIngestor(SpinBotIngestor):

    supported_measurements: ClassVar[list[str]] = ['spec_run']
    
    def make_spectra_plot(self, M, s, w):
        if len(M[s]) > 0:
            spectra = np.array(M[s])
            logger.info(f"spectra_shape={spectra.shape}")
            wls = np.array(M[w])
            for i in range(0, spectra.shape[0]):
                plt.plot(wls, spectra[i], label=f" spectra {i+1}")
            plt.legend()
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)
        return None

    def get_thumbnails(self):
        try:
            with h5py.File(self.file_to_upload, 'r') as h5file:
                M = h5file[f"measurement/{self.measurement}"]
                dtypes = [x.split("_")[0] for x in list(M.keys()) if x.endswith("spectra")]
                for dtype in dtypes:
                    img = self.make_spectra_plot(M, f'{dtype}_spectra', f'{dtype}_wls')
                    if img is not None:
                        self.add_thumbnail(img, f"SpinBot {dtype.upper()} Spectra")

                if 'photo' in list(M.keys()):
                    imarray = np.array(M['photo'])
                    self.add_thumbnail(Image.fromarray(imarray), "SpinBot SpecRun Image")
        except Exception as err:
            logger.error(f"failed to generate thumbnail for {self.file_to_upload} due to error {err}")


class SpinbotCameraCaptureIngestor(SpinBotIngestor):

    supported_measurements: ClassVar[list[str]] = ['zwo_camera_capture']
    
    def get_thumbnails(self):
        for format in ['jpg', 'tif']:
            try:
                image_file_name = f"{os.path.basename(self.file_to_upload)}.{format}"
                single_image = Image.open(image_file_name)
                self.add_thumbnail(single_image, f"ZWO Capture ({format})")
            except Exception as tnfail:
                logger.error(f"failed to generate thumbnail for {self.file_to_upload} due to error {tnfail}")
        

class SpinbotPhotoRunIngestor(SpinBotIngestor):
    supported_measurements: ClassVar[list[str]] = ['photo_run']


class SpinbotSpinRunIngestor(ScopeFoundryH5Ingestor):
    """Parses the SpinBot spin_run h5 file directly (replaces the yaml-based
    SpinRunIngestor_10kLegacy, which parsed the run metadata yaml). The separate
    deposition log yaml is unrelated and, as before, rides along unparsed since
    it carries no measurement/instrument_name tags.

    Parent dataset carries the full h5 tree and links only to the TRAY (batch)
    samples. Each thin film sample gets its own child dataset, linked only to
    that sample, carrying just that sample's own metadata — same parent/child
    split used by NirvanaMultiPosLineScanIngestor.
    """

    supported_measurements: ClassVar[list[str]] = ['spin_run']
    creation_location: ClassVar[str] = '67-4203'

    _SAMPLES_PATH = 'measurement/spin_run/samples'

    @staticmethod
    def _to_native(v):
        """Convert an h5py attribute value (numpy scalar/array) to a plain, JSON-serializable type."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return v

    def get_dataset_metadata(self):
        ScopeFoundryH5Ingestor.get_dataset_metadata(self)

        # spin_run's own run_id (assigned when the run starts) is the meaningful
        # identifier here, so it replaces ScopeFoundry's generic app-session
        self.unique_id = self.scientific_metadata['measurement']['spin_run']['settings']['run_id']
        self.parse_dataset_name()

        default_tags_value = "list,tags,separated,by,commas (optional)"
        default_session_value = "(optional)"

        try:
            scope_foundry_tags = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['tags'].strip()
            scope_foundry_session = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['session_name'].strip()
        except Exception:
            logger.warning("no mf-crucible settings found for tags or session_name")
            scope_foundry_tags = default_tags_value
            scope_foundry_session = default_session_value

        if scope_foundry_tags != default_tags_value:
            self.keywords += [x.strip() for x in scope_foundry_tags.split(",")]

        if scope_foundry_session != default_session_value:
            self.session_name = scope_foundry_session
            self.keywords += [self.session_name]

    def parse_orcid(self):
        if self.owner_orcid:
            return
        self.owner_orcid = check_orcid_entry(self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['orcid'])
        return

    def parse_project_id(self):
        if self.project_id:
            return
        self.project_id = self.scientific_metadata['hardware']['mf_crucible_spinbot']['settings']['proposal'].split(" ")[0]
        return

    def parse_measurement(self):
        self.measurement = 'spin_run'

    def parse_dataset_name(self):
        self.dataset_name = f"Spin Run - {self.unique_id[:13]}"

    def parse_data_type(self):
        self.data_type = 'Thin Film Deposition Run'

    def parse_samples(self):
        trays_seen = set()
        for tf_key in self.h5file[self._SAMPLES_PATH]:
            attrs = self.h5file[self._SAMPLES_PATH][tf_key].attrs
            tray_id = str(attrs['batch_id'])
            tray_name = str(attrs['batch_name'])
            sample_id = str(attrs['sample_id'])
            sample_name = str(attrs['sample_name'])

            # Tray (batch) — added once, linked to the parent dataset
            if tray_id not in trays_seen and _is_mfid(tray_id):
                trays_seen.add(tray_id)
                self.samples.append({
                    "unique_id": tray_id,
                    "sample_name": tray_name,
                    "sample_type": "spinbot tray",
                    "owner_orcid": self.owner_orcid,
                    "project_id": self.project_id,
                    "link_to_dataset": True,
                })

            # Thin film — not linked to parent dataset (linked at child dataset level)
            if not _is_mfid(sample_id):
                logger.info(f"skipping sample {tf_key}: invalid MFID {sample_id!r}")
                continue

            parent_ids = []
            if _is_mfid(tray_id):
                parent_ids.append(tray_id)
            precursor_mfid = str(attrs.get('precursor_mfid', ''))
            if _is_mfid(precursor_mfid):
                parent_ids.append(precursor_mfid)

            self.samples.append({
                "unique_id": sample_id,
                "sample_name": sample_name,
                "sample_type": "thin film",
                "owner_orcid": self.owner_orcid,
                "project_id": self.project_id,
                "parent_ids": parent_ids,
                "link_to_dataset": False,
            })
        return

    def parse_children(self):
        self.children = []
        for tf_key in self.h5file[self._SAMPLES_PATH]:
            attrs = self.h5file[self._SAMPLES_PATH][tf_key].attrs
            sample_id = str(attrs['sample_id'])
            sample_name = str(attrs['sample_name'])

            if not _is_mfid(sample_id):
                continue

            child_ds = Dataset(
                measurement='spin_run_tf',
                project_id=self.project_id,
                owner_orcid=self.owner_orcid,
                dataset_name=f"Spin Run for {sample_name} - {self.unique_id[:13]}",
                data_format=self.data_format,
                instrument_name=self.instrument_name,
                timestamp=self.timestamp,
            ).model_dump()
            child_md = {k: self._to_native(v) for k, v in attrs.items()}
            self.children.append({
                "dataset": child_ds,
                "scientific_metadata": child_md,
                "parent_id": self.unique_id,
                "sample_links": [sample_id],
            })
        return


class BioGlowIngestor(ScopeFoundryH5Ingestor):

    def is_file_supported(self):
        return self.file_to_upload.endswith('_bioglow_spec.h5')  


class QSpleemIngestor(ScopeFoundryH5Ingestor):
    """Base for QSpleem ingestors implementing the field convention:

        measurement -> human-readable name (e.g. 'LEEM-IV')
        data_type   -> 'ScopeFoundryH5.qspleem_<sf_group>.<subform>'

    The data_format prefix is hardcoded because ScopeFoundryH5Ingestor only sets
    self.data_format after parse_data_type() has already run.
    """
    _INSTRUMENT = 'qspleem'
    _GROUP = None            # ScopeFoundry measurement group, e.g. 'ARRES_EK'
    _LEED_THRESHOLD = 0.25   # median/p99 below this => diffraction (sparse frame)

    def _data_type(self, subform):
        return f"ScopeFoundryH5.{self._INSTRUMENT}_{self._GROUP.lower()}.{subform}"

    def _has_images(self):
        """True if this file carries a per-pixel diffraction stack."""
        try:
            return 'images' in self.h5file[f"measurement/{self._GROUP}"]
        except Exception:
            return False

    def _classify_plane(self, frame):
        """'diffraction' (LEED/SPLEED, sparse) or 'real_space' (LEEM/SPLEEM):
        a diffraction frame is mostly dark, so its median/p99 ratio is small."""
        frame = np.asarray(frame, dtype=np.float32)
        p99 = float(np.percentile(frame, 99)) or 1.0
        return 'diffraction' if float(np.median(frame)) / p99 < self._LEED_THRESHOLD else 'real_space'

    def _detect_imaging_mode(self, stack_path):
        """imaging_mode from a representative frame ~70% through an image stack;
        defaults to 'real_space' if the stack can't be read."""
        try:
            ds = self.h5file[stack_path]
            return self._classify_plane(ds[int(ds.shape[0] * 0.7)])
        except Exception:
            return 'real_space'

    def _add_diffraction_thumbnail(self, frame, caption):
        """Render one detector frame (p1-p99 grayscale) as a thumbnail."""
        frame = np.asarray(frame, dtype=np.float32)
        vmin, vmax = np.percentile(frame, 1), np.percentile(frame, 99)
        buf = BytesIO()
        plt.imsave(buf, frame, cmap='gray', format='png', origin='lower', vmin=vmin, vmax=vmax)
        buf.seek(0)
        self.add_thumbnail(Image.open(buf), caption)


class QSpleemImageIngestor(QSpleemIngestor):
    supported_measurements: ClassVar[list[str]] = ['image_save']
    _GROUP = 'image_save'

    def is_file_supported(self):
        return(self.file_to_upload.endswith('_image_save.h5'))

    @functools.cached_property
    def imaging_mode(self):
        try:
            M = self.h5file['measurement/image_save']
            key = next(k for k in M.keys() if 'im_array' in k)
            return self._classify_plane(M[key][()])
        except Exception:
            return 'real_space'

    def parse_measurement(self):
        self.measurement = 'LEED Image' if self.imaging_mode == 'diffraction' else 'LEEM Image'

    def parse_data_type(self):
        self.data_type = self._data_type(self.imaging_mode)

    def get_thumbnails(self):
        with h5py.File(self.file_to_upload, 'r') as h5file:
            M = h5file[f"measurement/image_save"]
            images = [k for k in list(M.keys()) if 'im_array' in k]
            buf = BytesIO()
            plt.imsave(buf, np.array(M[images[0]]), origin='lower', format='png')
            buf.seek(0)
            self.add_thumbnail(Image.open(buf), "Qspleem Image 0")


class QSpleemSVRampIngestor(QSpleemIngestor):
    supported_measurements: ClassVar[list[str]] = ['sv_ramp']
    _GROUP = 'sv_ramp'

    def is_file_supported(self):
        return self.file_to_upload.endswith('_sv_ramp.h5')

    @functools.cached_property
    def imaging_mode(self):
        return self._detect_imaging_mode('measurement/sv_ramp/000_im_array')

    def parse_measurement(self):
        self.measurement = 'LEED-IV' if self.imaging_mode == 'diffraction' else 'LEEM-IV'

    def parse_data_type(self):
        self.data_type = self._data_type(self.imaging_mode)

    def get_thumbnails(self):
        try:
            with h5py.File(self.file_to_upload, 'r') as h5file:
                M = h5file['measurement/sv_ramp']
                sv    = np.array(M['0000_sv_array'])
                imavg = np.array(M['000_imavg_array'])
                wfs   = float(M['0000_wfs_array'][0]) if '0000_wfs_array' in M else None
                has_images = '000_im_array' in M

            # ── IV curve ──────────────────────────────────────────────────
            fig, ax = plt.subplots()
            ax.plot(sv, imavg, color='tab:blue')
            if wfs is not None:
                ax.axvline(wfs, color='gray', linestyle='--', linewidth=1, label=f'WF {wfs:.2f} V')
                ax.legend(loc='upper right')
            ax.set_xlabel('Start Voltage (V)')
            ax.set_ylabel('Image intensity (a.u.)')
            buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
            self.add_thumbnail(Image.open(buf), 'SV Ramp IV Curve')

            # ── Image at max intensity after WF ───────────────────────────
            if has_images:
                if wfs is not None:
                    wf_idx = int(np.searchsorted(sv, wfs))
                    wf_idx = min(wf_idx, len(sv) - 2)
                else:
                    wf_idx = 0
                # Smooth and find bottom of the WF drop, then first peak after that
                smooth = np.convolve(imavg, np.ones(9)/9, mode="same")
                grad   = np.gradient(smooth, sv)
                sign_changes = np.where(np.diff(np.sign(grad[wf_idx:])) > 0)[0]
                min_idx = wf_idx + sign_changes[0] + 1 if len(sign_changes) > 0 else wf_idx + int(np.argmin(smooth[wf_idx:]))
                peak = min_idx + int(np.argmax(imavg[min_idx:]))
                with h5py.File(self.file_to_upload, 'r') as h5file:
                    frame = h5file['measurement/sv_ramp/000_im_array'][peak]
                fig, ax = plt.subplots()
                ax.imshow(frame, cmap='gray', origin='lower')
                ax.axis('off')
                wf_note = f' (post-WF {wfs:.2f} V)' if wfs is not None else ''
                ax.set_title(f'SV = {sv[peak]:.2f} V — max intensity{wf_note}', fontsize=8)
                buf = BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig); buf.seek(0)
                self.add_thumbnail(Image.open(buf), f'Image at SV {sv[peak]:.2f} V (max post-WF)')

        except Exception as e:
            logger.warning(f'SVRamp thumbnail generation failed: {e}')


def _arres_robust_vlim(values, k):
    """Linear vmin/vmax that emphasizes the low-intensity (material) signal.

    The bright specular regions are a high-value minority where we are not
    probing the material, so a median + k*MAD ceiling keeps the color range on
    the low-intensity structure and lets the specular saturate. MAD adapts per
    dataset, avoiding fragile fixed percentiles. Zeros (unmeasured points) are
    excluded from the statistics.
    """
    m = values[np.isfinite(values) & (values != 0)]
    if m.size == 0:
        return None, None
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med)))
    vmin = float(np.percentile(m, 2))
    vmax = med + k * 1.4826 * mad if mad > 0 else float(np.percentile(m, 98))
    return vmin, vmax


class QSpleemARRESEKIngestor(QSpleemIngestor):
    _GROUP = 'ARRES_EK'
    _MAD_K = 3

    def is_file_supported(self):
        # base + image companions: _ARRES_EK.h5, _ARRES_EK_images.h5, _ARRES_EK_images_0000.h5
        return bool(re.search(r'_ARRES_EK(_images(_\d+)?)?\.h5$', self.file_to_upload))

    def parse_measurement(self):
        self.measurement = 'ARRES E(k)'

    def parse_data_type(self):
        self.data_type = self._data_type('diffraction' if self._has_images() else 'spectrum')

    def plotEK(self, M, spec, E, uv):
        uvmin = f"({str(round(uv[0][0],2))}, {str(round(uv[0][1], 2))})"
        uvmax = f"({str(round(uv[-1][0], 2))}, {str(round(uv[-1][1], 2))})"
        fig, ax = plt.subplots()
        vmin, vmax = _arres_robust_vlim(spec, self._MAD_K)
        ax.imshow(spec, origin="lower", vmin=vmin, vmax=vmax)
        fig.set_size_inches(10, 10)
        ax.set_aspect('auto')
        ax.set_xlim([0, len(uv)-1])
        ax.set_xticks([0, len(uv)-1], [uvmin, uvmax])
        ax.set_yticks(range(0, len(E), 5), [round(x,1) for i,x in enumerate(E) if i % 5 == 0])
        ax.set_ylabel("Energy (eV)")
        ax.set_xlabel("uv")
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=400)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def get_thumbnails(self):
        with h5py.File(self.file_to_upload, 'r') as h5file:
            M = h5file["measurement/ARRES_EK"]
            if 'images' in M:
                # diffraction child: off-normal frame at 0.75 of max (u,v), at the highest energy
                spec = np.array(M['spectrum'])            # (spin, energy, k)
                E = np.array(M['eV'])
                uv = np.array(M['uv'])
                uv_max = uv[int(np.argmax(uv[:, 0] ** 2 + uv[:, 1] ** 2))]
                k_idx = int(np.argmin(np.sum((uv - 0.75 * uv_max) ** 2, axis=1)))
                e_idx = int(np.argmax(E))
                img = M['images']
                # Split image companions carry the FULL metadata (spectrum/eV/uv) but
                # only a slice of `images`; clamp the frame indices to what this file
                # actually holds so a representative frame is always in range — works
                # for any number of split parts.
                e_idx = min(e_idx, img.shape[1] - 1)
                k_idx = min(k_idx, img.shape[2] - 1)
                for spin in range(spec.shape[0]):
                    frame = img[spin, e_idx, k_idx]  # lazy (H, W) slice
                    self._add_diffraction_thumbnail(
                        frame, f"EK diffraction (spin {spin+1}, {E[e_idx]:.1f} eV, uv=({uv[k_idx][0]:.2f},{uv[k_idx][1]:.2f}))")
                return
            if not 'spectrum' in M.keys():
                return('no spectrum found')
            spec_series = np.array(M['spectrum'])
            E = np.array(M['eV'])
            uv = np.array(M['uv'])

        for i in range(0, spec_series.shape[0]):
            self.add_thumbnail(self.plotEK(M, spec_series[i, :, :], E, uv), f"QSpleem EK plot {i+1}")



class QSpleemARRESMMIngestor(QSpleemIngestor):
    _GROUP = 'ARRES_MM'
    _MAD_K = 3

    def is_file_supported(self):
        # base + image companions: _ARRES_MM.h5, _ARRES_MM_images.h5, _ARRES_MM_images_0000.h5
        return bool(re.search(r'_ARRES_MM(_images(_\d+)?)?\.h5$', self.file_to_upload))

    def parse_measurement(self):
        self.measurement = 'ARRES Constant Energy Surface'

    def parse_data_type(self):
        self.data_type = self._data_type('diffraction' if self._has_images() else 'momentum_map')

    def plotMM(self, spec, kx, ky, e):
        fig, ax = plt.subplots()
        vmin, vmax = _arres_robust_vlim(spec, self._MAD_K)
        disp = np.where(spec == 0, np.nan, spec)
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('lightgray')
        ax.imshow(disp, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_ylabel("ky")
        ax.set_xlabel("kx")
        ax.set_title(f"Energy: {e} eV")
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=400)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def get_thumbnails(self):
        with h5py.File(self.file_to_upload, 'r') as h5file:
            M = h5file["measurement/ARRES_MM"]
            if 'images' in M:
                # diffraction child: representative pattern at the 25th-percentile reflectivity point
                spec = np.array(M['spectrum'])            # (spin, kx, ky)
                img = M['images']
                # Split image companions carry the FULL metadata but only a slice of
                # `images`; clamp the frame indices to this file's actual shape so a
                # representative frame is always in range — works for any number of parts.
                for spin in range(spec.shape[0]):
                    s = spec[spin]
                    measured = s[np.isfinite(s) & (s != 0)]
                    if measured.size == 0:
                        continue
                    target = np.percentile(measured, 25)
                    s_masked = np.where(s == 0, np.nan, s)
                    idx = int(np.nanargmin(np.abs(s_masked - target)))
                    kxi, kyi = np.unravel_index(idx, s.shape)
                    kxi = min(int(kxi), img.shape[1] - 1)
                    kyi = min(int(kyi), img.shape[2] - 1)
                    frame = img[spin, kxi, kyi]   # lazy (H, W) slice
                    self._add_diffraction_thumbnail(frame, f"MM diffraction (spin {spin+1}, 25th pct reflectivity)")
                return
            spec_series = np.array(M['spectrum'])
            kx = np.array(M['kx'])
            ky = np.array(M['ky'])
            e = M['settings'].attrs['E']

        for i in range(0, spec_series.shape[0]):
            self.add_thumbnail(self.plotMM(spec_series[i, :, :], kx, ky, e), f"QSpleem Momentum Map {i+1}")


class NirvanaMultiPosLineScanIngestor(ScopeFoundryH5Ingestor):

    def is_file_supported(self):
        file_regex = r'.*pollux_oospec_multipos_line_scan.*\.h5'
        if re.match(file_regex, self.file_to_upload):
            return True
        else:
            return False
       # return(any([self.file_to_upload.endswith(f'_{x}.h5') for x in supported_measurements]))
    
    def get_dataset_metadata(self):
        self.instrument_name = 'Inorganic Nirvana'

        H5Ingestor.get_dataset_metadata(self)

        # overwrite unique ID if one is in the file
        if 'unique_id' in self.h5file.attrs.keys():
            self.unique_id = self.h5file.attrs['unique_id']

        # overwrite creation time and data format
        self.timestamp = datetime.fromtimestamp(self.h5file.attrs['time_id']).isoformat()
        self.data_format = "ScopeFoundryH5"

        # parse session_name and tags
        default_tags_value = "list,tags,separated,by,commas (optional)"
        default_session_value = "(optional)"

        try: 
            scope_foundry_tags = self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['tags'].strip()
            scope_foundry_session = self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['session_name'].strip()

        except Exception:
            logger.warning("no mf-crucible settings found for tags or session_name")
            scope_foundry_tags = default_tags_value
            scope_foundry_session = default_session_value

        if scope_foundry_tags != default_tags_value:
            self.keywords += [x.strip() for x in scope_foundry_tags.split(",")]

        if scope_foundry_session != default_session_value:
            self.session_name = scope_foundry_session
            self.keywords += [self.session_name]

    def parse_samples(self):
        trays_seen = set()
        pos_path = 'measurement/pollux_oospec_multipos_line_scan/positions'
        for pos in self.h5file[pos_path]:
            attrs = self.h5file[pos_path][pos].attrs
            sample_id = str(attrs['sample_uuid'])
            sample_name = str(attrs['sample_name'])
            tray_id = str(attrs['tray_uuid'])
            tray_name = str(attrs['tray_name'])

            # Add tray once per unique valid MFID — linked to parent dataset
            if tray_id not in trays_seen and _is_mfid(tray_id):
                trays_seen.add(tray_id)
                self.samples.append({
                    "unique_id": tray_id,
                    "sample_name": tray_name,
                    "owner_orcid": self.owner_orcid,
                    "project_id": self.project_id,
                    "link_to_dataset": True,
                })

            # Thin film — not linked to parent dataset (linked at child dataset level)
            if not _is_mfid(sample_id):
                logger.info(f"skipping position {pos}: invalid MFID '{sample_id}'")
                continue

            sample = {
                "unique_id": sample_id,
                "sample_name": sample_name,
                "owner_orcid": self.owner_orcid,
                "project_id": self.project_id,
                "link_to_dataset": False,
            }
            if _is_mfid(tray_id):
                sample["parent_ids"] = [tray_id]
            self.samples.append(sample)
        return

    def parse_children(self):
        pos_path = 'measurement/pollux_oospec_multipos_line_scan/positions'
        for pos in self.h5file[pos_path]:
            attrs = self.h5file[pos_path][pos].attrs
            sample_id = str(attrs['sample_uuid'])
            sample_name = str(attrs['sample_name'])

            if not _is_mfid(sample_id):
                continue

            child_ds = Dataset(
                measurement=self.measurement,
                project_id=self.project_id,
                owner_orcid=self.owner_orcid,
                dataset_name=f"Child Nirvana scan for {sample_name}",
                data_format=self.data_format,
                instrument_name=self.instrument_name,
                timestamp=self.timestamp,
            ).model_dump()
            child_md = {
                "integration_time": float(attrs['integration_time']),
                "x_center": float(attrs['x_center']),
                "y_center": float(attrs['y_center']),
                "x_positions": attrs['x_positions'].tolist(),
                "y_positions": attrs['y_positions'].tolist(),
            }
            split_path = self._create_split_h5(pos)
            self.children.append({
                "dataset": child_ds,
                "scientific_metadata": child_md,
                "parent_id": self.unique_id,
                "sample_links": [sample_id],
                "files_to_upload": [split_path],
            })
        return

    def _create_split_h5(self, pos_key):
        from pathlib import Path
        os.makedirs(TMP_DIR, exist_ok=True)
        stem = Path(self.file_to_upload).stem
        out_path = os.path.join(TMP_DIR, f"{stem}_{pos_key}.h5")
        pos_path = 'measurement/pollux_oospec_multipos_line_scan/positions'
        wl_path  = 'measurement/pollux_oospec_multipos_line_scan/wavelengths'
        with h5py.File(self.file_to_upload, 'r') as src, h5py.File(out_path, 'w') as dst:
            for group in ('app', 'hardware'):
                if group in src:
                    src.copy(group, dst)
            meas = dst.require_group('measurement/pollux_oospec_multipos_line_scan')
            src.copy(wl_path, meas, name='wavelengths')
            meas_src = src['measurement/pollux_oospec_multipos_line_scan']
            if 'settings' in meas_src:
                src.copy(f'measurement/pollux_oospec_multipos_line_scan/settings', meas, name='settings')
            pos_grp = meas.require_group('positions')
            src.copy(f'{pos_path}/{pos_key}', pos_grp, name=pos_key)
        return out_path

    def parse_orcid(self):
        if self.owner_orcid:
            return
        self.owner_orcid = check_orcid_entry(self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['orcid'])
        return

    def parse_project_id(self):
        if self.project_id:
            return
        else:
             self.project_id = self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['project'].split(" ")[0]
        return

    def setup_data(self):
        self._child_position = self._detect_child_position()
        if self._child_position is not None:
            self._setup_as_from_holders_child()
        else:
            super().setup_data()

    def _detect_child_position(self):
        """
        Return the position string (e.g. 'S03') if this dataset is a from-holders
        child, or None otherwise.

        'position' is written into the dataset record by the uploader at create time,
        before ingestion runs — making it the only timing-safe discriminator. Hierarchy
        checks (list_parents, include_links) cannot be used here because the parent-child
        dataset link and sample links are created by the uploader after create_dataset()
        returns (i.e., after this ingestor has already finished).
        """
        try:
            ds = get_client().datasets.get(self.unique_id, include_metadata=True)
        except requests.exceptions.HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                return None
            raise

        raw_scimd = ds.get('scientific_metadata', {})
        if isinstance(raw_scimd, dict) and 'scientific_metadata' in raw_scimd:
            actual_scimd = raw_scimd['scientific_metadata']
        else:
            actual_scimd = raw_scimd or {}

        return actual_scimd.get('position')

    def _setup_as_from_holders_child(self):
        """Run ingestion for a from-holders child dataset."""
        self.get_scientific_metadata()
        self.get_dataset_metadata()
        self.get_acl_information()
        self._apply_child_scientific_metadata()
        self.thumbnails = []
        self._add_spectrum_thumbnail()

    def _position_key(self, position):
        """Convert position string (e.g. 'S03') to the h5 group key at that index."""
        pos_path = 'measurement/pollux_oospec_multipos_line_scan/positions'
        idx = int(position[1:]) - 1
        return list(self.h5file[pos_path].keys())[idx]

    def _apply_child_scientific_metadata(self):
        pos_path = 'measurement/pollux_oospec_multipos_line_scan/positions'
        pos_key = self._position_key(self._child_position)
        attrs = self.h5file[pos_path][pos_key].attrs
        self.scientific_metadata = {
            "integration_time": float(attrs['integration_time']),
            "x_center": float(attrs['x_center']),
            "y_center": float(attrs['y_center']),
            "x_positions": attrs['x_positions'].tolist(),
            "y_positions": attrs['y_positions'].tolist(),
        }

    def _add_spectrum_thumbnail(self):
        try:
            pos_path = 'measurement/pollux_oospec_multipos_line_scan/positions'
            pos_key = self._position_key(self._child_position)
            pos = self.h5file[pos_path][pos_key]

            wls = np.array(self.h5file['measurement/pollux_oospec_multipos_line_scan/wavelengths'])
            raw = np.array(pos['raw_intensities'])     # (N_scans, N_wl)
            dark = np.array(pos['dark_intensities'])   # (N_wl,)
            blank = np.array(pos['blank_intensities']) # (N_wl,)

            denom = blank - dark
            denom = np.where(np.abs(denom) < 1, 1.0, denom)
            T = (raw.mean(axis=0) - dark) / denom
            absorbance = -np.log10(T.clip(1e-9))

            fig, ax = plt.subplots()
            ax.plot(wls, absorbance)
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Absorbance")
            ax.set_title(f"Position {self._child_position}")
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            plt.close(fig)
            buf.seek(0)
            self.add_thumbnail(Image.open(buf), f"Absorbance Spectrum ({self._child_position})")
        except Exception as err:
            logger.error(f"Failed to generate spectrum thumbnail for position {self._child_position}: {err}")


class NirvanaMultiPosSpecRunIngestor(ScopeFoundryH5Ingestor):

    _MEAS_PATH = 'measurement/pollux_multipos_spec_run'

    def is_file_supported(self):
        return bool(re.match(r'.*pollux_multipos_spec_run.*\.h5', self.file_to_upload))

    def get_dataset_metadata(self):
        self.instrument_name = 'Inorganic Nirvana'

        H5Ingestor.get_dataset_metadata(self)

        if 'unique_id' in self.h5file.attrs.keys():
            self.unique_id = self.h5file.attrs['unique_id']

        if 'time_id' in self.h5file.attrs:
            self.timestamp = datetime.fromtimestamp(self.h5file.attrs['time_id']).isoformat()
        self.data_format = "ScopeFoundryH5"

        default_tags_value = "list,tags,separated,by,commas (optional)"
        default_session_value = "(optional)"

        try:
            scope_foundry_tags = self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['tags'].strip()
            scope_foundry_session = self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['session_name'].strip()
        except Exception:
            logger.warning("no mf-crucible settings found for tags or session_name")
            scope_foundry_tags = default_tags_value
            scope_foundry_session = default_session_value

        if scope_foundry_tags != default_tags_value:
            self.keywords += [x.strip() for x in scope_foundry_tags.split(",")]

        if scope_foundry_session != default_session_value:
            self.session_name = scope_foundry_session
            self.keywords += [self.session_name]

    def parse_orcid(self):
        if self.owner_orcid:
            return
        self.owner_orcid = check_orcid_entry(
            self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['orcid'])

    def parse_project_id(self):
        if self.project_id:
            return
        self.project_id = self.scientific_metadata['hardware']['mf_crucible_nirvana']['settings']['project'].split(" ")[0]

    def _measure_flags(self):
        settings = self.h5file[f'{self._MEAS_PATH}/settings']
        has_uvvis = bool(settings.attrs.get('measure_uvvis', True))
        has_pl = bool(settings.attrs.get('measure_pl', False))
        return has_uvvis, has_pl

    def _primary_pos_path(self):
        has_uvvis, _ = self._measure_flags()
        if has_uvvis:
            return f'{self._MEAS_PATH}/uvvis/positions'
        return f'{self._MEAS_PATH}/pl/positions'

    def parse_samples(self):
        trays_seen = set()
        pos_path = self._primary_pos_path()
        for pos in self.h5file[pos_path]:
            attrs = self.h5file[pos_path][pos].attrs
            sample_id = str(attrs['sample_uuid'])
            sample_name = str(attrs['sample_name'])
            tray_id = str(attrs['tray_uuid'])
            tray_name = str(attrs['tray_name'])

            if tray_id not in trays_seen and _is_mfid(tray_id):
                trays_seen.add(tray_id)
                self.samples.append({
                    "unique_id": tray_id,
                    "sample_name": tray_name,
                    "owner_orcid": self.owner_orcid,
                    "project_id": self.project_id,
                    "link_to_dataset": True,
                })

            if not _is_mfid(sample_id):
                logger.info(f"skipping position {pos}: invalid MFID '{sample_id}'")
                continue

            sample = {
                "unique_id": sample_id,
                "sample_name": sample_name,
                "owner_orcid": self.owner_orcid,
                "project_id": self.project_id,
                "link_to_dataset": False,
            }
            if _is_mfid(tray_id):
                sample["parent_ids"] = [tray_id]
            self.samples.append(sample)

    def parse_children(self):
        has_uvvis, has_pl = self._measure_flags()
        pos_path = self._primary_pos_path()

        for pos in self.h5file[pos_path]:
            attrs = self.h5file[pos_path][pos].attrs
            sample_id = str(attrs['sample_uuid'])
            sample_name = str(attrs['sample_name'])

            if not _is_mfid(sample_id):
                continue

            child_md = {"position": pos}
            if has_uvvis:
                uvvis_attrs = self.h5file[f'{self._MEAS_PATH}/uvvis/positions/{pos}'].attrs
                child_md.update({
                    "uvvis_integration_time": float(uvvis_attrs['integration_time']),
                    "x_center": float(uvvis_attrs['x_center']),
                    "y_center": float(uvvis_attrs['y_center']),
                    "x_positions": uvvis_attrs['x_positions'].tolist(),
                    "y_positions": uvvis_attrs['y_positions'].tolist(),
                })
            if has_pl:
                pl_attrs = self.h5file[f'{self._MEAS_PATH}/pl/positions/{pos}'].attrs
                child_md.update({
                    "pl_integration_time": float(pl_attrs['integration_time']),
                    "pl_x_center": float(pl_attrs['x_center']),
                    "pl_y_center": float(pl_attrs['y_center']),
                })

            child_ds = Dataset(
                measurement=f"{self.measurement}_tf",
                project_id=self.project_id,
                owner_orcid=self.owner_orcid,
                dataset_name=f"Nirvana SpecRun for {sample_name}",
                data_format=self.data_format,
                instrument_name=self.instrument_name,
                timestamp=self.timestamp,
            ).model_dump()

            split_path = self._create_split_h5(pos)
            self.children.append({
                "dataset": child_ds,
                "scientific_metadata": child_md,
                "parent_id": self.unique_id,
                "sample_links": [sample_id],
                "files_to_upload": [split_path],
            })

    def _create_split_h5(self, pos_key):
        from pathlib import Path
        os.makedirs(TMP_DIR, exist_ok=True)
        stem = Path(self.file_to_upload).stem
        out_path = os.path.join(TMP_DIR, f"{stem}_{pos_key}.h5")
        has_uvvis, has_pl = self._measure_flags()
        with h5py.File(self.file_to_upload, 'r') as src, h5py.File(out_path, 'w') as dst:
            meas = dst.require_group(self._MEAS_PATH)
            meas_src = src[self._MEAS_PATH]
            if 'settings' in meas_src:
                src.copy(f'{self._MEAS_PATH}/settings', meas, name='settings')
            if has_uvvis:
                uvvis_grp = meas.require_group(f'uvvis_{pos_key}')
                src_uvvis = src[f'{self._MEAS_PATH}/uvvis']
                src_uvvis_pos = src_uvvis[f'positions/{pos_key}']
                for k, v in src_uvvis_pos.attrs.items():
                    uvvis_grp.attrs[k] = v
                src.copy(src_uvvis['wavelengths'], uvvis_grp, name='wavelengths')
                for name, item in src_uvvis_pos.items():
                    src.copy(item, uvvis_grp, name=name)
            if has_pl:
                pl_grp = meas.require_group(f'pl_{pos_key}')
                src_pl = src[f'{self._MEAS_PATH}/pl']
                src_pl_pos = src_pl[f'positions/{pos_key}']
                for k, v in src_pl_pos.attrs.items():
                    pl_grp.attrs[k] = v
                for name in ('wavelengths', 'dark_intensities', 'reference_intensities'):
                    if name in src_pl:
                        src.copy(src_pl[name], pl_grp, name=name)
                for name, item in src_pl_pos.items():
                    src.copy(item, pl_grp, name=name)
        return out_path

    def setup_data(self):
        self._child_position = self._detect_child_position()
        if self._child_position is not None:
            self._setup_as_from_holders_child()
        else:
            super().setup_data()

    def _detect_child_position(self):
        try:
            ds = get_client().datasets.get(self.unique_id, include_metadata=True)
        except requests.exceptions.HTTPError as err:
            if err.response is not None and err.response.status_code == 404:
                return None
            raise
        raw_scimd = ds.get('scientific_metadata', {})
        if isinstance(raw_scimd, dict) and 'scientific_metadata' in raw_scimd:
            actual_scimd = raw_scimd['scientific_metadata']
        else:
            actual_scimd = raw_scimd or {}
        return actual_scimd.get('position')

    def _setup_as_from_holders_child(self):
        self.get_scientific_metadata()
        self.get_dataset_metadata()
        self.get_acl_information()
        self._apply_child_scientific_metadata()
        self.thumbnails = []
        self._add_child_thumbnails()
        self.measurement = 'Nirvana_Child_SpecRun'

    def _position_key(self, position):
        pos_path = self._primary_pos_path()
        pos_keys = list(self.h5file[pos_path].keys())
        if position in pos_keys:
            return position
        idx = int(position[1:]) - 1
        return pos_keys[idx]

    def _apply_child_scientific_metadata(self):
        has_uvvis, has_pl = self._measure_flags()
        pos_key = self._position_key(self._child_position)
        is_split = (f'{self._MEAS_PATH}/uvvis_{self._child_position}' in self.h5file
                    or f'{self._MEAS_PATH}/pl_{self._child_position}' in self.h5file)
        child_md = {}
        if has_uvvis:
            if is_split:
                uvvis_grp = self.h5file[f'{self._MEAS_PATH}/uvvis_{self._child_position}']
            else:
                pos_key = self._position_key(self._child_position)
                uvvis_grp = self.h5file[f'{self._MEAS_PATH}/uvvis/positions/{pos_key}']
            uvvis_attrs = uvvis_grp.attrs
            child_md.update({
                "uvvis_integration_time": float(uvvis_attrs['integration_time']),
                "x_center": float(uvvis_attrs['x_center']),
                "y_center": float(uvvis_attrs['y_center']),
                "x_positions": uvvis_attrs['x_positions'].tolist(),
                "y_positions": uvvis_attrs['y_positions'].tolist(),
            })
        if has_pl:
            if is_split:
                pl_grp = self.h5file[f'{self._MEAS_PATH}/pl_{self._child_position}']
            else:
                pos_key = self._position_key(self._child_position)
                pl_grp = self.h5file[f'{self._MEAS_PATH}/pl/positions/{pos_key}']
            pl_attrs = pl_grp.attrs
            child_md.update({
                "pl_integration_time": float(pl_attrs['integration_time']),
                "pl_x_center": float(pl_attrs['x_center']),
                "pl_y_center": float(pl_attrs['y_center']),
            })
        self.scientific_metadata = child_md

    def _add_child_thumbnails(self):
        has_uvvis, has_pl = self._measure_flags()
        is_split = (f'{self._MEAS_PATH}/uvvis_{self._child_position}' in self.h5file
                    or f'{self._MEAS_PATH}/pl_{self._child_position}' in self.h5file)

        if has_uvvis:
            try:
                if is_split:
                    grp = self.h5file[f'{self._MEAS_PATH}/uvvis_{self._child_position}']
                    wls = np.array(grp['wavelengths'])
                else:
                    pos_key = self._position_key(self._child_position)
                    wls = np.array(self.h5file[f'{self._MEAS_PATH}/uvvis/wavelengths'])
                    grp = self.h5file[f'{self._MEAS_PATH}/uvvis/positions/{pos_key}']
                raw = np.array(grp['raw_intensities'])
                dark = np.array(grp['dark_intensities'])
                blank = np.array(grp['blank_intensities'])
                denom = blank - dark
                denom = np.where(np.abs(denom) < 1, 1.0, denom)
                T = (raw.mean(axis=0) - dark) / denom
                absorbance = -np.log10(T.clip(1e-9))
                fig, ax = plt.subplots()
                ax.plot(wls, absorbance)
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("Absorbance")
                ax.set_title(f"Position {self._child_position}")
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                plt.close(fig)
                buf.seek(0)
                self.add_thumbnail(Image.open(buf), f"UVVis Absorbance ({self._child_position})")
            except Exception as err:
                logger.error(f"Failed to generate UVVis thumbnail for position {self._child_position}: {err}")

        if has_pl:
            try:
                if is_split:
                    grp = self.h5file[f'{self._MEAS_PATH}/pl_{self._child_position}']
                    wls = np.array(grp['wavelengths'])
                    dark_shared = np.array(grp['dark_intensities'])
                else:
                    pos_key = self._position_key(self._child_position)
                    wls = np.array(self.h5file[f'{self._MEAS_PATH}/pl/wavelengths'])
                    dark_shared = np.array(self.h5file[f'{self._MEAS_PATH}/pl/dark_intensities'])
                    grp = self.h5file[f'{self._MEAS_PATH}/pl/positions/{pos_key}']
                raw = np.array(grp['raw_intensities'])
                intensity = raw.mean(axis=0) - dark_shared
                fig, ax = plt.subplots()
                ax.plot(wls, intensity)
                ax.set_xlabel("Wavelength (nm)")
                ax.set_ylabel("PL Intensity (counts)")
                ax.set_title(f"Position {self._child_position}")
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=150)
                plt.close(fig)
                buf.seek(0)
                self.add_thumbnail(Image.open(buf), f"PL Intensity ({self._child_position})")
            except Exception as err:
                logger.error(f"Failed to generate PL thumbnail for position {self._child_position}: {err}")


class QSpleemSVRampSpinIngestor(QSpleemIngestor):
    supported_measurements: ClassVar[list[str]] = ['sv_ramp_spin']
    _GROUP = 'sv_ramp_spin'

    def is_file_supported(self):
        return self.file_to_upload.endswith('_sv_ramp_spin.h5')

    @functools.cached_property
    def imaging_mode(self):
        return self._detect_imaging_mode('measurement/sv_ramp_spin/000_im_up_array')

    def parse_measurement(self):
        self.measurement = 'SPLEED-IV' if self.imaging_mode == 'diffraction' else 'SPLEEM-IV'

    def parse_data_type(self):
        self.data_type = self._data_type(self.imaging_mode)

    def get_thumbnails(self):
        try:
            with h5py.File(self.file_to_upload, 'r') as h5file:
                M = h5file['measurement/sv_ramp_spin']
                sv         = np.array(M['0000_sv_array'])
                imavg_up   = np.array(M['000_imavg_up_array'])
                imavg_down = np.array(M['000_imavg_down_array'])
                asym       = np.array(M['000_asym_array'])
                emga_up    = np.array(M['000_emga_up_array'])   if '000_emga_up_array'   in M else None
                emga_down  = np.array(M['000_emga_down_array']) if '000_emga_down_array' in M else None
                has_images = '000_im_up_array' in M

            # ── IV curves: spin up + spin down ────────────────────────────
            fig, ax = plt.subplots()
            ax.plot(sv, imavg_up,   color='tab:blue', label='Spin Up')
            ax.plot(sv, imavg_down, color='tab:red',  label='Spin Down')
            ax.set_xlabel('Start Voltage (V)')
            ax.set_ylabel('Mean Intensity (counts)')
            ax.legend(loc='upper right')
            buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
            self.add_thumbnail(Image.open(buf), 'SV Ramp Spin IV Curves')

            # ── Asymmetry vs SV ───────────────────────────────────────────
            fig, ax = plt.subplots()
            ax.plot(sv, asym, color='tab:green')
            ax.set_xlabel('Start Voltage (V)')
            ax.set_ylabel('Asymmetry')
            buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
            self.add_thumbnail(Image.open(buf), 'Spin Asymmetry vs SV')

            # ── Emission current (if present) ─────────────────────────────
            if emga_up is not None or emga_down is not None:
                fig, ax = plt.subplots()
                if emga_up   is not None: ax.plot(sv, emga_up,   color='tab:blue', label='Spin Up')
                if emga_down is not None: ax.plot(sv, emga_down, color='tab:red',  label='Spin Down')
                ax.set_xlabel('Start Voltage (V)')
                ax.set_ylabel('Emission Current (A)')
                ax.legend(loc='upper right')
                buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                self.add_thumbnail(Image.open(buf), 'GaAs Emission Current vs SV')

            # ── 3-panel image at max |asymmetry| ─────────────────────────
            if has_images:
                window = 5
                asym_smooth = np.convolve(np.abs(asym), np.ones(window) / window, mode='same')
                peak = int(np.argmax(asym_smooth))
                with h5py.File(self.file_to_upload, 'r') as h5file:
                    M = h5file['measurement/sv_ramp_spin']
                    up   = M['000_im_up_array'][peak].astype(np.float32)
                    down = M['000_im_down_array'][peak].astype(np.float32)
                total = up + down
                asym_frame = np.where(total > 0, (up - down) / total, 0.0)

                def norm_gray(arr):
                    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
                    return np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(norm_gray(up),   cmap='gray',   origin='lower'); axes[0].set_title('Spin Up');   axes[0].axis('off')
                axes[1].imshow(norm_gray(down), cmap='gray',   origin='lower'); axes[1].set_title('Spin Down'); axes[1].axis('off')
                v = float(np.percentile(np.abs(asym_frame), 98)) or 1.0
                axes[2].imshow(asym_frame, cmap='RdBu_r', origin='lower', vmin=-v, vmax=v)
                axes[2].set_title('Asymmetry'); axes[2].axis('off')
                fig.suptitle(f'SV = {sv[peak]:.2f} V (max |asymmetry|)', fontsize=10)
                plt.tight_layout()
                buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                self.add_thumbnail(Image.open(buf), f'Images at SV {sv[peak]:.2f} V (max |asym|)')

        except Exception as e:
            logger.warning(f'SVRampSpin thumbnail generation failed: {e}')


class QSpleemSPLEEMImageIngestor(QSpleemIngestor):
    supported_measurements: ClassVar[list[str]] = ['SPLEEM_image']
    _GROUP = 'SPLEEM_image'

    def is_file_supported(self):
        return self.file_to_upload.endswith('_SPLEEM_image.h5')

    def parse_measurement(self):
        self.measurement = 'SPLEEM Image'

    def parse_data_type(self):
        self.data_type = self._data_type('real_space')

    def get_thumbnails(self):
        try:
            with h5py.File(self.file_to_upload, 'r') as h5file:
                images = np.array(h5file['measurement/SPLEEM_image/images'], dtype=np.float32)
        except Exception as e:
            logger.warning(f'Could not read SPLEEM_image data: {e}')
            return

        avg_up   = images[:, 0, :, :].mean(axis=0)
        avg_down = images[:, 1, :, :].mean(axis=0)
        total    = avg_up + avg_down
        asym     = np.where(total > 0, (avg_up - avg_down) / total, 0.0)

        def to_img_gray(arr):
            lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
            arr = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
            buf = BytesIO()
            plt.imsave(buf, arr, cmap='gray', format='png', origin='lower')
            buf.seek(0)
            return Image.open(buf)

        def to_img_rdbu(arr):
            v = np.percentile(np.abs(arr), 98) or 1.0
            buf = BytesIO()
            plt.imsave(buf, arr, cmap='RdBu_r', vmin=-v, vmax=v, format='png', origin='lower')
            buf.seek(0)
            return Image.open(buf)

        self.add_thumbnail(to_img_gray(avg_up),   'SPLEEM Spin Up (averaged)')
        self.add_thumbnail(to_img_gray(avg_down), 'SPLEEM Spin Down (averaged)')
        self.add_thumbnail(to_img_rdbu(asym),     'SPLEEM Asymmetry (averaged)')


class QSpleemDepositionMonitorIngestor(QSpleemIngestor):
    supported_measurements: ClassVar[list[str]] = ['deposition_monitor']
    _GROUP = 'deposition_monitor'

    def is_file_supported(self):
        return self.file_to_upload.endswith('_deposition_monitor.h5')

    def parse_measurement(self):
        self.measurement = 'Deposition Monitor'

    def parse_data_type(self):
        self.data_type = self._data_type('time_series')

    def get_thumbnails(self):
        ROI_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

        try:
            # Images are saved on their own (tunable) interval, so the image
            # stack and the continuously-sampled monitoring arrays are on
            # different axes / lengths. Only the first and last frames are
            # needed here, so read them lazily rather than the whole stack.
            with h5py.File(self.file_to_upload, 'r') as f:
                M = f['measurement/deposition_monitor']
                imgs = M['images']
                spin_mode = imgs.ndim == 4   # (N, 2, H, W) vs (N, H, W)
                first_frame = np.asarray(imgs[0, 0] if spin_mode else imgs[0], dtype=np.float32)
                last_up     = np.asarray(imgs[-1, 0] if spin_mode else imgs[-1], dtype=np.float32)
                last_down   = np.asarray(imgs[-1, 1], dtype=np.float32) if spin_mode else None
                roi_times    = np.array(M['roi_times'])
                roi_int      = np.array(M['roi_intensity'])
                roi_pos      = np.array(M['roi_positions'])
                ec           = np.array(M['emission_current'])
                pressure     = np.array(M['pressure_main_chamber'])
                temperature  = np.array(M['sample_temperature'])

            # Infer a scalar time axis for ec/pressure/temperature.
            # roi_times has shape (N,) or (N, 2); use column 0 for axis.
            t = roi_times[:, 0] if roi_times.ndim == 2 else roi_times
            n_roi = roi_int.shape[-1]

            def _frame_with_rois(frame, roi_row, caption):
                lo, hi = np.percentile(frame, 2), np.percentile(frame, 98)
                norm = np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)
                fig, ax = plt.subplots()
                ax.imshow(norm, cmap='gray', origin='lower')
                for i in range(n_roi):
                    x, y, w, h = roi_row[i]
                    rect = plt.Rectangle((x, y), w, h, linewidth=1.5,
                                         edgecolor=ROI_COLORS[i % len(ROI_COLORS)], facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x + w/2, y + h + 10, f'ROI {i+1}',
                            color=ROI_COLORS[i % len(ROI_COLORS)], ha='center', fontsize=7)
                ax.axis('off')
                ax.set_title(caption, fontsize=8, pad=4)
                buf = BytesIO(); plt.savefig(buf, format='png', dpi=150, bbox_inches='tight'); plt.close(fig); buf.seek(0)
                self.add_thumbnail(Image.open(buf), caption)

            # ── ROI intensity vs time ─────────────────────────────────────
            try:
                if n_roi > 0:
                    fig, ax = plt.subplots()
                    if spin_mode:
                        t_up   = roi_times[:, 0]
                        t_down = roi_times[:, 1]
                        for i in range(n_roi):
                            c = ROI_COLORS[i % len(ROI_COLORS)]
                            ax.plot(t_up,   roi_int[:, 0, i], color=c, linestyle='-',  label=f'ROI {i+1} Up')
                            ax.plot(t_down, roi_int[:, 1, i], color=c, linestyle='--', label=f'ROI {i+1} Down')
                    else:
                        for i in range(n_roi):
                            ax.plot(t, roi_int[:, i], color=ROI_COLORS[i % len(ROI_COLORS)], label=f'ROI {i+1}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Intensity (counts)')
                    ax.legend(fontsize=7)
                    buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                    self.add_thumbnail(Image.open(buf), 'ROI Intensity vs Time')
            except Exception as e:
                logger.warning(f'DepositionMonitor ROI intensity thumbnail failed: {e}')

            # ── ROI asymmetry vs time (spin mode only) ───────────────────
            try:
                if spin_mode and n_roi > 0:
                    fig, ax = plt.subplots()
                    t_up, t_down = roi_times[:, 0], roi_times[:, 1]
                    for i in range(n_roi):
                        up, dn = roi_int[:, 0, i], roi_int[:, 1, i]
                        total = up + dn
                        asym = np.where(total > 0, (up - dn) / total, np.nan)
                        ax.plot(t_up, asym, color=ROI_COLORS[i % len(ROI_COLORS)], label=f'ROI {i+1}')
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Asymmetry')
                    ax.legend(fontsize=7)
                    buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                    self.add_thumbnail(Image.open(buf), 'ROI Asymmetry vs Time')
            except Exception as e:
                logger.warning(f'DepositionMonitor ROI asymmetry thumbnail failed: {e}')

            # ── First frame with first ROIs / final frame with final ROIs ─
            try:
                _frame_with_rois(first_frame, roi_pos[0], 'First Frame with ROIs')
            except Exception as e:
                logger.warning(f'DepositionMonitor first-frame thumbnail failed: {e}')
            try:
                _frame_with_rois(last_up, roi_pos[-1], 'Final Frame with ROIs')
            except Exception as e:
                logger.warning(f'DepositionMonitor final-frame thumbnail failed: {e}')

            # ── Scalar time series — combined ─────────────────────────────
            try:
                fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                for ax, arr, ylabel in zip(axes,
                    [ec, pressure, temperature],
                    ['Emission Current (A)', 'Pressure (mbar)', 'Temperature (°C)']):
                    ax.plot(t, arr, color='tab:blue', linewidth=1)
                    ax.set_xlabel('Time (s)'); ax.set_ylabel(ylabel)
                plt.tight_layout()
                buf = BytesIO(); plt.savefig(buf, format='png', dpi=150); plt.close(fig); buf.seek(0)
                self.add_thumbnail(Image.open(buf), 'Instrument Parameters vs Time')
            except Exception as e:
                logger.warning(f'DepositionMonitor instrument-parameters thumbnail failed: {e}')

            # ── Final frame asymmetry (spin mode only) ────────────────────
            try:
                if spin_mode:
                    total = last_up + last_down
                    asym_frame = np.where(total > 0, (last_up - last_down) / total, 0.0)
                    v = float(np.percentile(np.abs(asym_frame), 98)) or 1.0
                    buf = BytesIO()
                    plt.imsave(buf, asym_frame, cmap='RdBu_r', vmin=-v, vmax=v, format='png', origin='lower')
                    buf.seek(0)
                    self.add_thumbnail(Image.open(buf), 'Final Frame Asymmetry')
            except Exception as e:
                logger.warning(f'DepositionMonitor final-frame-asymmetry thumbnail failed: {e}')

        except Exception as e:
            logger.warning(f'DepositionMonitor thumbnail generation failed: {e}')












        









