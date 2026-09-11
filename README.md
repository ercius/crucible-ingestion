# crucible-ingestion

Python classes and framework for parsing metadata out of scientific data files and into
the Crucible Data Platform.

Parsing and sending are separate. `parse()` reads a file and returns an
`IngestionPacket` describing everything that would be sent; `push_packet()` takes that
packet and writes it to Crucible. Nothing is uploaded during parsing, and the file bytes
themselves are never sent — only the parsed metadata.

## Install

```
uv sync
```

## Authentication

Run `crucible config init` once to store your API url and key. Everything in this package
builds its client through `get_client()`, which picks up that config.

## Running

A real run needs a `--dsid`. Create the dataset in Crucible first. You may want to include
fields that you do not expect to be parsable, such as instrument name or project ID:

```python
client.datasets.create(Dataset())
```

Then pass the id it returns:

```
uv run crucible-ingest --file {local_file_path} --dsid {dsid} [--ingestor {ingestion-class}] [--output-dir {output_dir}] [--push]
```

- `--dsid` is the dataset the parsed information belongs to.
- `--ingestor` names an ingestion class explicitly. Without it, a supported ingestor is
  auto-detected by `find_supported_ingestor`.
- `--output-dir` defaults to `./dry_run_output/{dsid}`.
- `--push` sends the parsed packet to Crucible. Without it nothing is written to Crucible
  and the run is a dry run.

Running `--push` with neither a `--dsid` nor `--test` is an error.

The parsed packet is written to `{output_dir}/packet.json` and any thumbnails are decoded
to `{output_dir}/thumbnails/` so they can be inspected.

`parse()` reads from Crucible when a dataset record with the given dsid already exists, so
that values a user set at dataset creation are not overwritten by parsed ones. If no such
record exists, parsing starts from scratch.

## Testing

Testing does not need a dsid. Without one an mfid is generated locally, and `--test`
creates a throwaway dataset under that id in the `crucible-test` project for `--push` to
write to, logging the id it picked:

```
uv run crucible-ingest --file {local_file_path} --push --test
```

Drop `--push` to parse only. The packet is written to `{output_dir}/packet.json` under the
generated id and nothing touches Crucible:

```
uv run crucible-ingest --file {local_file_path}
```

Passing a `--dsid` is optional and useful when you want the run to go against a dataset
that already exists — to check that `parse()` reads the user provided fields back
correctly, for instance. With a `--dsid`, `--test` creates nothing and the push goes to
that dataset.

## Using it from Python

This requires an dataset id. Use the mfid package.

```python
from crucible_ingestion import parse, push_packet
from mfid import mfid

packet = parse(path_to_file, dsid=mfid()[0])
if packet is not None:
    # inspect packet.dataset_fields, packet.scientific_metadata,
    # packet.keywords, packet.samples, packet.children, packet.thumbnails
    push_packet(packet)
```

`parse` returns `None` when no ingestor supports the file.

## Adding a new ingestor

Ingestor classes live in `crucible_ingestion/ingestors/` and subclass
`CrucibleDatasetIngestor` (`ingestors/crucible_ingestor.py`), which defines the parsing
hooks `setup_data()` calls: `get_scientific_metadata`, `get_dataset_metadata`,
`get_acl_information`, `parse_batch`, `parse_samples`, `parse_children`, and
`get_thumbnails`.

A new class needs `is_file_supported()` and must be added to `ingestor_list` in
`crucible_ingestion/ingestors/registry.py`. Order matters — `find_supported_ingestor`
returns the first class in the list that claims the file, so specific ingestors belong
above general ones.

Parsing must not post to Crucible. All writes to Crucible should be part of the `push_packet` function. 
Anything that needs to be created or linked can be added to the packet during parsing, for example: 
`self.samples`, `self.children`, `self.keywords`, or `self.thumbnails`.

Once you are content with your new parser, add an example file to `tests/data` and generate
its schema from the tests folder:

```
uv run python tests/make_schemas.py --file tests/data/{new_file}
```

This runs `parse()` and records the packet's key paths, types, and values as a JSON Schema
in `tests/schemas/`, paired with the data file's hash in `tests/schema_registry.json`. The
test suite checks later parses against it, so a field that stops being parsed shows up as a
failure. Files already in the registry are skipped, so running it with no arguments picks
up only what is new. Review the generated schema before committing — it records what the
code does today, bugs included.

## Test data

The files the tests parse live in GCS, not in this repo. Pull them before running the
tests:

```
gcloud storage rsync -r gs://crucible-ingestion-test-data tests/data
```

Push new or updated files back up:

```
gcloud storage rsync -r tests/data gs://crucible-ingestion-test-data
```

Both directions only transfer what is missing or changed, and neither deletes anything at
the destination. Removing a file from the bucket has to be done explicitly.

## Running in the cloud

The RabbitMQ consumer that runs this package in GCP lives in a separate repo,
`ingestion-cloud-consumer`.
