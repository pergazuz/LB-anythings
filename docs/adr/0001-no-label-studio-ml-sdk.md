---
status: accepted
---
# Do not depend on the `label-studio-ml` SDK

Label Studio documents ML backends as subclasses of the SDK's `LabelStudioMLBase`
served by the SDK's Flask app and job-directory machinery. We implement the HTTP
contract ourselves on FastAPI and depend on the SDK for nothing. The previous
backend had already abandoned the SDK's server because its job-directory scanning
produced constant benign tracebacks; the only pieces still used were two helpers
(parse the label config into tag names, download a task image with the access
token), which are a few dozen lines to own. A framework base class has no place in
a hexagonal core.

## Consequences

- Our media resolver handles Label Studio upload paths, local-files paths, plain
  http(s) URLs and local files. It does not handle s3:// or gs:// references, which
  the SDK's resolver did and which we never used.
- The wire protocol (`/health`, `/setup`, `/predict`, `/webhook`, `/train`,
  `/is_training`, `/metrics`, `/versions`) is ours to keep compatible by test.
