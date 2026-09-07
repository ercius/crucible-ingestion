import os
import yaml
import logging

from .crucible_ingestor import CrucibleDatasetIngestor
from crucible.models import Dataset
from ..client import get_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SpinRunIngestor_10kLegacy(CrucibleDatasetIngestor):

    def is_file_supported(self):
        # currently expects yaml file
        if not self.file_to_upload.endswith('yaml'):
            return False
        
        # open the file
        with open(self.file_to_upload, 'r') as f:
            file_content = yaml.safe_load(f)
        
        # check that measurement = spin_run
        if file_content.get('measurement') != 'spin_run':
            return False
        
        # second check for instrument
        if file_content.get('instrument_name') != 'SpinBot':
            return False
        
        run_id = file_content.get('run_id')
        if run_id != self.unique_id:
            raise Exception(f'Run ID in file does not match Crucible Dataset MFID. {run_id=} and {self.unique_id=}')
        
        self.file_contents = file_content
        return True


    def get_scientific_metadata(self):
        self.scientific_metadata = self.file_contents


    def parse_sample_metadata(self, sample_mfid):
        logger.info(sample_mfid)
        samples = self.file_contents.get('samples')
        logger.info([s['sample_id'] for s in samples])
        additional_info = {k:v for k,v in self.file_contents.items() if k != 'samples'}
        sample_info = next(s for s in samples if s['sample_id'] == sample_mfid)
        return {**sample_info, **additional_info}


    def parse_measurement(self):
        self.measurement = self.file_contents.get('measurement')
        logger.info(f"{self.measurement=}")
    

    def parse_dataset_name(self):
        run_id = self.file_contents.get('run_id', None)
        self.dataset_name = f'Spin Run - {run_id[0:13]}'

    
    def parse_file_timestamp(self):
        self.file_timestamp = self.file_contents.get('creation_time', None)


    def parse_instrument(self):
        self.instrument_name = self.file_contents.get('instrument_name')
        logger.info(f"{self.instrument_name=}")
    

    def parse_data_type(self):
        self.data_type = 'Thin Film Deposition Run'
    

    def parse_samples(self):
        parent_list = []
        sample_list = self.file_contents.get('samples', [])
        for s in sample_list:
            # IDs are required: without them samples cannot be deduplicated on re-ingest
            sample_id = s.get('sample_id')
            batch_id = s.get('batch_id')
            if not sample_id:
                raise ValueError(f"spin run sample {s.get('sample_name')!r} is missing 'sample_id'")
            if not batch_id:
                raise ValueError(f"spin run sample {s.get('sample_name')!r} is missing 'batch_id'")

            # collect the basic sample info
            sample = {
                'unique_id': sample_id,
                'sample_name': s.get('sample_name'),
                'sample_type': 'thin film',
                'owner_orcid': self.owner_orcid,
                'project_id': self.project_id,
                'parent_ids': [],
                'link_to_dataset': True
            }

            # samples have 'batch' (tray) parents
            if not batch_id in parent_list:
                parent_list.append(batch_id)
                batch_sample = {
                    'unique_id': batch_id,
                    'sample_name': s.get('batch_name'),
                    'sample_type': 'spinbot tray',
                    'owner_orcid': self.owner_orcid,
                    'project_id': self.project_id,
                    'parent_ids': [],
                    'link_to_dataset': False
                }

                # add batch to sample list to make sure it gets created
                self.samples.append(batch_sample)

            sample['parent_ids'].append(batch_id)

            # samples have precursor parents
            # precursors need to get ID's in instrument control software
            precursor_mfid = s.get('precursor_mfid')
            precursor_name = s.get('precursor_solution_name')
            if precursor_mfid:
                sample['parent_ids'].append(precursor_mfid)

            elif precursor_name:
                found_ps = get_client().samples.list(sample_name = precursor_name,
                                               project_id = self.project_id)
                # only resolve by name when it is unambiguous
                if len(found_ps) == 1:
                    sample['parent_ids'].append(found_ps[-1]['unique_id'])

                elif len(found_ps) > 1:
                    raise Exception(f'Precursor Unique ID not provided and multiple samples with name found in this project: {found_ps}')

                else:
                    logger.info(f'Precursor ID not provided and no sample named {precursor_name!r} found in this project')

            # add sample to list to be created
            self.samples.append(sample)

        return


    def parse_children(self):
        self.children = []
        for sample in self.file_contents.get('samples', []):
            sample_name = sample['sample_name']
            sample_id = sample['sample_id']
            child_ds_name = f'Spin Run for {sample_name} - {self.unique_id[0:13]}'
            child_ds = Dataset(
                        measurement = self.measurement,
                        project_id = self.project_id,
                        owner_orcid = self.owner_orcid,
                        dataset_name = child_ds_name,
                        data_format = 'yaml' ).model_dump()

            md = self.parse_sample_metadata(sample['sample_id'])
            self.children.append({'dataset': child_ds,
                             'scientific_metadata': md,
                             'parent_id': self.unique_id,
                             'sample_links':[sample_id]})

    def parse_orcid(self):
        if self.owner_orcid:
            return
        self.owner_orcid = self.file_contents.get('user_orcid', None)


    def parse_project_id(self):
        if self.project_id:
            return
        self.project_id = self.file_contents.get('project_id', None)



    
