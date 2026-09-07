import logging
from .scope_foundry_ingestors import ( SimpleTiledImageScopeFoundryH5Ingestor,
                                                BioGlowIngestor,
                                                QSpleemSVRampIngestor,
                                                QSpleemSVRampSpinIngestor,
                                                QSpleemImageIngestor,
                                                QSpleemDepositionMonitorIngestor,
                                                QSpleemSPLEEMImageIngestor,
                                                QSpleemARRESEKIngestor,
                                                QSpleemARRESMMIngestor,
                                                CanonCaptureScopeFoundryH5Ingestor,
                                                SingleSpecScopeFoundryH5Ingestor,
                                                HyperspecScopeFoundryH5Ingestor,
                                                HyperspecSweepScopeFoundryH5Ingestor,
                                                ToupcamLiveScopeFoundryH5Ingestor,
                                                CLSyncRasterScanIngestor,
                                                CLHyperspecIngestor,
                                                SpinbotSpecLineIngestor,
                                                SpinbotSpecRunIngestor,
                                                SpinbotCameraCaptureIngestor,
                                                SpinbotPhotoRunIngestor,
                                                SpinbotSpinRunIngestor,
                                                NirvanaMultiPosLineScanIngestor,
                                                NirvanaMultiPosSpecRunIngestor,
                                                ScopeFoundryH5Ingestor)
from .rga_tey_batch_ingestor import RgaTeyBatchIngestor
from .gc_log_ingestor import GCLogIngestor
from .biologic_mpt_ingestor import BiologicMptIngestor
from .image_ingestor import ImageIngestor, TifIngestor
from .insitu_pl_ingestor import InSituPlIngestor
from .dm_ingestor import DigitalMicrographIngestor
from .emi_ingestor import EmiIngestor
from .ser_ingestor import SerIngestor
from .bcf_ingestor import BcfIngestor
from .emd_ingestor import BerkeleyEmdIngestor
from .emd_velox_ingestor import VeloxEmdIngestor
from .jupiter_afm_ingestor import AFMIngestor
from .czi_ingestor import CziIngestor
from .ptychography_h5_ingestor import PtychographyH5Ingestor
from .h5_ingestor import H5Ingestor
from .autobot_spinrun_yaml_ingestor import SpinRunIngestor_10kLegacy
from .inorganic_xrd_ingestor import InorganicXRDIngestor


logger = logging.getLogger(__name__)
logger.info("imported all classes")

ingestor_list = [AFMIngestor,
                PtychographyH5Ingestor,
                SimpleTiledImageScopeFoundryH5Ingestor, 
                BioGlowIngestor,
                QSpleemSVRampIngestor,
                QSpleemSVRampSpinIngestor,
                QSpleemImageIngestor,
                QSpleemSPLEEMImageIngestor,
                QSpleemDepositionMonitorIngestor,
                QSpleemARRESEKIngestor,
                QSpleemARRESMMIngestor,
                RgaTeyBatchIngestor,
                CanonCaptureScopeFoundryH5Ingestor, 
                SingleSpecScopeFoundryH5Ingestor,
                HyperspecScopeFoundryH5Ingestor,
                HyperspecSweepScopeFoundryH5Ingestor,
                ToupcamLiveScopeFoundryH5Ingestor,
                CLSyncRasterScanIngestor,
                CLHyperspecIngestor, 
                SpinbotSpecLineIngestor,
                SpinbotCameraCaptureIngestor, 
                SpinbotPhotoRunIngestor,
                SpinbotSpinRunIngestor,
                SpinRunIngestor_10kLegacy,
                InorganicXRDIngestor,
                InSituPlIngestor,
                CziIngestor,
                DigitalMicrographIngestor,
                EmiIngestor,
                SerIngestor,
                BcfIngestor,
                BerkeleyEmdIngestor,
                VeloxEmdIngestor,
                SpinbotSpecRunIngestor,
                ImageIngestor,
                TifIngestor,
                BiologicMptIngestor,
                GCLogIngestor,
                NirvanaMultiPosSpecRunIngestor,
                NirvanaMultiPosLineScanIngestor,
                ScopeFoundryH5Ingestor,
                H5Ingestor] 

SELECTABLE = {cls.__name__: cls for cls in ingestor_list} 

def find_supported_ingestor(dataset_to_process,
                            dsid,
                            specified_ingestor = None,
                            ingestor_list = ingestor_list):
    
    if specified_ingestor is not None:
        cls = SELECTABLE.get(specified_ingestor)
        if cls is None:
            logger.warning(f"Specified ingestor '{specified_ingestor}' not found, falling back to list scan")
        else:
            logger.info(cls)
            ig = cls(file_to_upload = dataset_to_process, unique_id = dsid)
            if ig.is_file_supported():
                logger.info(f"{dataset_to_process} is supported by {specified_ingestor}")
                return ig, specified_ingestor
            else:
                logger.warning(f"{dataset_to_process} not supported by {specified_ingestor}")

    # if that ingestor class was not supported, check the others
    for ingestor_class in ingestor_list:
        ig = ingestor_class(file_to_upload = dataset_to_process, unique_id = dsid)

        if ig.is_file_supported():
            logger.info(f"{dataset_to_process} is supported by {ingestor_class.__name__}")
            return ig, ingestor_class.__name__
        else:
            continue

    return None, None