gollek/runner/native — native runner helper docs

Canonical SafeTensors implementation

This repository includes a full SafeTensors implementation under:

  gollek/runner/safetensor/

Use that module for SafeTensors loading, conversion to GGUF, sharded models, and inference integrations. Avoid duplicating SafeTensors parsing/loader logic in other modules — prefer the existing gollek-safetensor-* modules.

Native adapter

A small adapter module was added to provide a programmatic bridge for native callers:

  gollek/runner/native/gollek-native-adapters

Usage (preferred, CDI): inject tech.kayys.gollek.safetensor.loader.SafetensorLoaderFacade and create an instance of
tech.kayys.gollek.nativeadapter.NativeSafeTensorsAdapter.of(facade)

Usage (programmatic): pass a SafetensorLoaderFacade instance to the adapter constructor and call loadCached(Path) or open(Path).

Notes

- The adapter intentionally delegates to the existing safetensor modules — it does not duplicate parsing logic.
- If you want a purely static convenience API, consider wiring a Provider that the adapter can use; for now the adapter is intentionally simple.
