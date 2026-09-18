# Verified KML/KMZ repair

Status: **Implemented and verified locally on Windows; macOS packaged validation remains a release gate.** September 17, 2026.

Delivery evidence and platform release gates are tracked in the
[implementation report](docs/validation/kml-repair-implementation.md).

This revision incorporates independent parser, UI/lifecycle, and export reviews.
The audit found a reproducible document-selection hazard, snapshot-lifetime
contradictions, and missing recovery/export requirements. Those corrections are
part of the requirements below, not optional follow-up work. Revision 3 adds a
verified catalog of narrowly permitted formatting/import corrections, combined
defect handling, and actionable client replacement requests. It does not promise
recovery of missing geometry or coverage of every exporter defect.

## 1. Recommendation and verified finding

Add **Repair & analyze** to the invalid-input workflow. The action automatically
creates a private effective source snapshot, verifies the permitted edit, and continues
analysis only when verification succeeds. Preserve the original file. Implement
the four explicit rules below, each with its own proof and refusal conditions.

The supplied `C:\Users\rbake\Downloads\Pipelines August 2023.kmz` is repairable
without changing its shape geometry:

| Inspection | Verified result |
| --- | --- |
| Archive | 34,475 bytes; one entry, `doc.kml`; ZIP CRC check passed |
| Original KML | 174,966 bytes; UTF-8 |
| Failure | `unbound prefix: line 6, column 1` |
| Cause | Five `Document` attributes use `xsi:schemaLocation`, but no `xmlns:xsi` declaration exists |
| Repair | Insert ` xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"` into the root start tag |
| Extent | One 54-byte insertion at zero-based byte offset 211; no replacements or deletions |
| Strict XML parsing | Passes after the insertion |
| Application parsing | 42 source pipelines, 44 separate LineString paths, 2,139 vertices, no points |
| Parser diagnostics | Only the ordinary informational primary-document selection message |
| Geometry preservation | All 44 coordinate blocks byte-identical; raw path order and coordinates match application output |
| Original mileage after repair | 920,747.8695152604 meters / 572.1250571784351 US survey miles |
| Original file | Unchanged, verified by byte comparison and SHA-256 |

The original archive SHA-256 is
`92bea274a0e588e2611e4a46b96b8a8f48d3b26f9089533a1b9dfbbd89f1eb4c`.
Removing the inserted bytes recovers the exact original KML, including altitude
values, geometry structure, names, IDs, styles, and metadata. This is stronger
evidence than merely comparing rounded mileage.

Evidence: [inspection report](docs/validation/kml-repair-inspection.json) and
[reproducible investigation](scripts/validation/inspect_august_2023_repair.py).
The investigation used an automatically deleted temporary KMZ. It did **not** run
overlap analysis, state analysis, or full KML schema validation. The mileage above
is a measured result of the repaired input, not a successfully analyzed baseline
from the invalid original. That initial planning investigation did not implement
the production repair engine or rebuild the EXE; subsequent delivery evidence is
recorded separately in the implementation report linked above.

XML requires declared prefixes; `xsi:schemaLocation` belongs to the XML Schema
instance vocabulary. The rule restores that metadata namespace without fetching
any schema. References: [W3C namespace constraints](https://www.w3.org/TR/xml-names/#nsc-NSDeclared),
[W3C schema-instance attributes](https://www.w3.org/TR/xmlschema-1/#Instance_Document_Constructions).

## 2. Version 1 repair catalog and refusal rules

Only the missing `xsi` case has been observed in the supplied customer file.
The other additions have standards support and reproducible synthetic evidence;
their frequency in client files has not been measured. Do not advertise this as
an exhaustive repair tool. Common XML mistakes include incorrect tag case and
incomplete tags, but those do not establish an unambiguous original structure.
[Google's KML FAQ](https://developers.google.com/kml/faq).

| Rule ID | Permitted correction | Protected content |
| --- | --- | --- |
| `missing_xsi_schema_namespace_v1` | Add one missing, proven metadata namespace declaration | Every existing byte remains unchanged |
| `leading_xml_whitespace_v1` | Remove only XML whitespace before a valid leading XML declaration | BOM, declaration, and entire document body unchanged |
| `literal_metadata_ampersand_v1` | Encode a confirmed literal `&` as `&amp;` in the restricted text leaves below | Identical displayed literal text; all geometry, IDs, links, and markup unchanged |
| `filename_format_mismatch_v1` | Route valid KML/KMZ content through its actual format; save with the correct suffix | Zero content edits for this rule; original filename/file unchanged |

### A. Missing schema-instance declaration

Allow the declaration insertion only when all of these conditions hold:

1. A bounded, strictly decoded UTF-8 document, optionally with a UTF-8 BOM, has a
   KML 2.2 root and demonstrably undeclared `xsi` metadata attributes. An earlier
   allowlisted defect may mask the unbound-prefix error in the first parser pass.
2. A bounded XML structural reader identifies real element/attribute tokens while respecting
   quoted attributes, comments, CDATA, and processing instructions. It confirms
   the root start-tag insertion point and balanced document structure. For an
   otherwise well-formed document, standard-library Expat with namespace processing
   disabled can inspect it, with a quote-aware byte locator for the root start
   tag. Combined defects require the bounded transaction described below; Expat
   alone still rejects bare ampersands and misplaced declarations. Strict
   namespace-aware parsing remains mandatory after repair. Do not implement a
   permissive replacement XML grammar. Do not use
   a global regular expression or split on the first `>` inside a quoted value.
3. All uses of the missing `xsi` prefix are `xsi:schemaLocation` or
   `xsi:noNamespaceSchemaLocation` metadata attributes on KML `Document`/`Folder`
   elements. No `xsi` element names, geometry attributes, `xsi:type`, `xsi:nil`,
   unknown prefixed attributes, or unresolved QName-valued constructs qualify.
4. The reader checks namespace scope throughout the document and resolves the
   root/container names to their expanded KML namespace names. Existing,
   contradictory, shadowed, or otherwise ambiguous `xsi` declarations fail this
   conservative rule; never overwrite a declaration. Decode namespace attribute
   character references according to XML rules; compare namespace URIs exactly.
5. No DTD/entity declarations, signed XML, ambiguous encoding, or non-allowlisted
   syntax damage is present. Reject DTD declarations before entity expansion, not only
   after a namespace parse failure. No external entity resolution, schema fetch,
   or XInclude expansion is permitted. A BOM, XML declaration, and actual UTF-8
   decoding must agree. These are unsupported repair cases, not excuses to weaken
   strict parsing. Offsets are byte offsets even when earlier text is multibyte.
   Standard predefined references (`&amp;`, `&lt;`, etc.) and valid numeric
   character references remain allowed; they are not DTD/custom entities.
6. This rule makes exactly one root-attribute insertion. The complete candidate,
   including any other independently eligible edits, must strictly parse.

### B. Whitespace before the XML declaration

Allow deletion of a nonempty run consisting **only** of space, tab, CR, and LF
between byte zero (or one leading UTF-8 BOM) and the sole leading `<?xml ...?>`
declaration. Require a valid, consistent XML 1.0 UTF-8 declaration, an otherwise
eligible complete KML document, and preservation of the BOM and all bytes from
the declaration onward, except other independently recorded catalog edits.

This fixes the observed parser error `XML or text declaration not at start of
entity`. XML's prolog grammar requires the declaration first and defines this
specific whitespace set. [W3C XML prolog](https://www.w3.org/TR/xml/#sec-prolog-dtd).

Do not strip comments, processing instructions, duplicate BOMs/declarations,
NULs, non-ASCII spaces, email/export banners, Markdown fences, or arbitrary
prefix/suffix data. Whitespace before a root with **no** declaration and trailing
XML whitespace already work; leave them unchanged. Do not trim text inside KML.

### C. Literal ampersands in simple display metadata

Examples: `<name>A & B</name>` and `<Snippet>Survey &</Snippet>`.
Permit replacement of that one literal byte with `&amp;` only when:

- The actual expanded name is KML `name`, `description`, or `Snippet`, directly
  beneath a KML `Document`, `Folder`, or `Placemark`. Namespace scope and feature
  membership are unambiguous. A similarly named foreign element does not qualify.
- The entire leaf is plain character data plus already-valid predefined/numeric
  references, with complete matching tags and no child markup, CDATA, comments,
  processing instructions, or raw `<` inside its text. Other leaves containing
  valid CDATA/comments are preserved, not rewritten.
- The literal ampersand is immediately followed by XML whitespace or this leaf's
  closing tag. This deliberately excludes ambiguous `A&B`, `&custom;`, `&nbsp;`,
  unfinished `&amp`, and malformed `&#...` references. Do not guess an entity's
  intended expansion. Existing valid references must remain byte-identical.
- Only the ampersand encoding changes. Independently decode the restricted
  original leaf's valid references, treat the explicitly permitted literal
  ampersands as text, and compare to the strict candidate's text after normal
  XML newline handling. Keep Unicode spelling/normalization and whitespace.

No escaping in attributes, geometry, coordinates, tracks, timestamps, identifiers,
ExtendedData, styles, links, resource URLs, or arbitrary unknown elements. Do not
wrap a damaged description in CDATA, run an HTML sanitizer, fix broken HTML, or
escape an entire file. Plain display text remains the same: `A & B` is not renamed
to a different pipeline. [XML character data](https://www.w3.org/TR/xml/#syntax),
[KML description escaping](https://developers.google.com/kml/documentation/kmlreference#description).

### D. Correct format, wrong filename extension

Handle UTF-8 plaintext KML named `.kmz`, and a complete ZIP/KMZ named `.kml`, as
an import-routing correction. Only the top-level selected file is eligible; do
not rewrite link targets or archive member suffixes. Require a bounded,
unambiguous recognized format, a KML root, and the same full verification and
coverage checks as other repairs. A `PK` signature or `is_zipfile()` result alone
is insufficient. Refuse polyglots, self-extracting preambles, appended unexplained
data, corrupt archives, and ZIPs without supported KML. KMZ is a ZIP container;
do not convert between containers to retain an incorrect suffix.
[Google KMZ documentation](https://developers.google.com/kml/documentation/kmzarchives).

- Snapshot before detection. Record supplied suffix and verified effective format;
  route explicitly from that snapshot without renaming or rereading the source.
  XML-edit rules may combine with this rule through the transaction below.
- For ZIP content, validate all members and freeze the existing primary-member
  selection and dependency graph before any XML edits. Keep member names intact.
- For misnamed plaintext KML, initially require self-contained content under the
  complete base/reference definition in section 5. Ambiguous relative links or
  self-filename dependencies need a correctly exported client file. Ordinary
  correctly named standalone KML retains the planned linked-source behavior.
- The success notice says **“File type recognized. Geometry verified unchanged.
  Original preserved.”** when this is the sole correction; Details explains the
  extension mismatch. The optional copy uses `.kml` or `.kmz` for its effective
  format. Pure format correction saves byte-for-byte identical file content.
  Independently reopen using the corrected suffix before publishing the copy.

### One bounded transaction, not repeated guesses

Support both standalone KML and KMZ documents. Other encodings remain supported
by ordinary parsing where they work today; automatic repair initially declines
them rather than transcodes. Valid UTF-16, one UTF-8 BOM, valid entities/CDATA,
line endings, and harmless whitespace are not defects. Neither the supplied-file
investigation nor the fixed synthetic patches are production lexical scanners.

Treat `eligible` as permission to attempt the explicit allowlist, not a
promise of repair success. The detector may discover additional errors during
verification. Produce **one** deterministic candidate per source snapshot, batching
all independently eligible edits across reachable documents. Apply non-overlapping
patches defined against original byte offsets, then validate the whole result.
No partial success, incremental analysis, automatic second guessing pass, or
repeated repair prompts for individual defects. Any unresolved defect blocks
automatic continuation. Reopening a verified saved copy is `not_needed`.

The lexical inspector must identify source boundaries without forgiving malformed
structure. Use a bounded, quote/comment/CDATA/PI-aware token locator, declining
uncertain contexts; let the standard strict parser enforce XML grammar. After
strict parsing, an independent verifier must map candidate element/text boundaries
back through the patch ledger to original bytes and recheck every rule and
protected region. Do not accept a detector's claimed context on trust. Namespace-
disabled Expat is an inspection aid only where the original syntax permits it.
No global regex replacement or error-line-only matching qualifies as detection.

Bound proposed edits to 10,000 patches and 1 MiB aggregate inserted/deleted bytes
per source snapshot, in addition to section 5's size/work limits. Limit exhaustion
is **unsupported by this repair version**, not evidence that the client's geometry
is corrupt. Record complete patch evidence within these limits; UI previews may
summarize it. Validate idempotence and invert every patch to exact source bytes.

Evidence: [catalog probe report](docs/validation/kml-repair-catalog-probes.json)
and [fixed synthetic investigation](scripts/validation/inspect_repair_catalog.py).
Twelve KML/KMZ patch probes, two byte-identical format corrections, five already-
valid controls, and seven refusal observations ran against the current parser.
Patched examples preserve full geometry subtrees (including altitude, path breaks,
points, polygon rings, and tracks) and match authored valid pipeline records and
unrounded mileage. These prove the mechanisms, **not** a production detector,
prevalence, overlap/state equivalence, or completion of release acceptance tests.

### Explicitly outside initial automatic repair

| Problem | Required behavior |
| --- | --- |
| Unclosed/mismatched tags or truncated document | Refuse; missing geometry or boundaries cannot be reconstructed reliably |
| Invalid, missing, non-finite, or out-of-range coordinates | Refuse geometry repair; do not remove vertices, swap axes, clamp, interpolate, or reconnect lines |
| Invalid polygon rings, duplicate vertices, zero-length segments | Preserve as supplied; do not simplify, deduplicate, close, or run `make_valid` |
| Missing `gx`, `kml`, default KML, or another geometry namespace | Refuse; familiar prefixes or a schemaLocation URL do not establish the missing binding; never strip prefixes or normalize namespace URLs |
| Ampersands outside rule C, unknown/malformed entities, broken HTML/CDATA, raw `<`, invalid control characters | Refuse; do not delete characters, invent entity expansions, or reinterpret potential markup |
| Incorrect element case, duplicate/conflicting attributes, wrong tag names | Refuse; do not promote, merge, or rename potentially different geometry structures |
| Guessed/misdeclared encoding, duplicate BOM, arbitrary preamble/trailer, multiple roots | Refuse; no lossy decode, encoding-label rewrite, concatenation, or truncation at the first closing root |
| Corrupt/encrypted ZIP, missing KML, duplicate/ambiguous archive paths | Report the container problem; do not select a different dataset or salvage a partial archive |
| Missing/remote linked documents | Preserve existing diagnostics; never fabricate or download content to claim repair success |
| XML Base, XInclude, or ambiguous NetworkLink structure in a repaired graph | Refuse automatic repaired analysis while their geometry/dependency semantics cannot be verified; disabling Save alone is insufficient |
| Invalid IDs or style references | Preserve; renaming or deleting them is outside the catalog |

Do not use permissive XML recovery (`recover=True`), silently strip namespace
prefixes, or reserialize an entire document as the repair mechanism.

## 3. What “verified” means

Verification has separate stages and statuses. Parsing successfully is necessary
but does not alone establish preserved geometry or complete analysis.

1. **Source snapshot:** Read a bounded immutable snapshot. Record archive and
   per-document hashes. The detector, repairer, verifier, and analyzer must consume
   the same snapshot, avoiding edits to the source between inspection and use.
   Freeze original primary-member selection, reachable link targets, stable
   traversal order, and dependency identities as well as their bytes. Freeze
   reachable standalone files as they are discovered; validate acquisition again
   before offering the repair decision. A hash manifest identifies the bytes
   consumed, not an atomic point-in-time snapshot of independently changing files.
   Detect changes during acquisition and restart explicitly or decline; never
   silently combine newly read dependencies with an approved snapshot.
2. **Edit proof:** Record rule/version, member path, original byte offset, exact
   removed/inserted text, before/after hashes, and reason. Independently validate
   each permitted location and context. Inverting all patches must recover the
   exact original bytes. Everything outside those explicitly permitted spans is
   unchanged. Format-only routing records zero byte edits and identical content
   hashes. Record original-to-candidate offset mappings for accurate diagnostics.
3. **Geometry protection:** The lexical inventory preserves ordered geometry
   structure, feature membership, path breaks, every coordinate/altitude token,
   tracks/timestamps, IDs, and links. Include points and polygon geometry even
   though this app does not count polygon outlines as pipeline mileage. Compare
   supported path counts/order/vertices against the application's documented
   extraction projection; do not merge separate paths into one line. Keep two
   distinct checks:
   - **Source preservation:** exact original bytes outside the approved spans, ordered
     element/container inventory, raw coordinate/altitude tokens, and references.
   - **Application projection:** current parsing emits LineStrings before
     `gx:Track` paths within a Placemark, uses longitude/latitude for measurement,
     prioritizes lines over points, and stores point summaries without coordinates.
     Match those rules explicitly. Raw mixed-geometry document order is not
     universally identical to normalized path order. Preserve altitude bytes;
     do not claim the application analyzes altitude or all geometry types.
   Foreign-namespace elements using reserved KML geometry names, incorrectly
   cased reserved geometry/feature elements, nested Placemarks, and other ambiguous
   feature membership must block verified automatic analysis rather than exploit
   current local-name matching or silently ignored elements. Report suspicious
   spelling without automatically interpreting it as geometry. The catalog probe
   demonstrates that lowercase `linestring` can parse yet lose expected paths.
   The independent inventory must actively detect case-insensitive reserved-name
   lookalikes such as `placemark`, `linestring`, `coordinates` in a wrongly cased
   parent, or `gx:track` in its declared extension namespace. Parser diagnostic
   codes alone cannot provide this guard. Unknown coordinate-bearing structures
   with ambiguous extraction semantics also block automatic repaired analysis;
   documented non-mileage geometry such as KML polygons remains preserved under
   the explicit projection above.
   Validate NetworkLink element namespaces, child cardinality and href structure
   independently. Reject XML Base and XInclude in repaired sources, including
   otherwise valid linked documents: unchanged inline coordinates do not prove
   that the intended external geometry was selected. Never ask the client simply
   to remove a base/include declaration; request a complete correctly resolved export.
4. **Strict parsing:** Run the ordinary strict XML parser on every repaired member,
   then run the application parser on the effective source. This is application
   compatibility validation, **not** a claim of universal KML XSD validity or
   real-world positional accuracy. KML has multiple geometry types and coordinate
   semantics: [KML reference](https://developers.google.com/kml/documentation/kmlreference).
5. **Completeness check:** No fatal XML failures, missing required linked
   documents, rejected supported geometries, or unexplained geometry inventory
   differences may remain before automatic continuation. Informational or
   unrelated warnings remain visible. Classify issues by typed codes and scope,
   not display text or severity alone. Repair blockers include `invalid_coordinate`,
   invalid/short tracks or lines, `malformed_placemark`, `linked_kml_parse_error`,
   unresolved/missing/remote/out-of-base/unsupported NetworkLinks, and unexplained
   inventory loss. Enumerate the exact set beside the parser's existing
   `INCOMPLETE_CODES`; cover warning-level ignored geometry separately.
   Unsupported polygon-only or point-only input cannot become a successful
   pipeline run. A document may have verified preserved bytes but still be
   blocked from automatic analysis; keep that distinction in the report.
   This stricter complete-input gate governs repaired candidates. Ordinary inputs
   keep existing partial-analysis/diagnostic behavior, apart from the explicitly
   documented reader safety-policy changes.
6. **Normal analysis:** Use the existing analyzer, overlap settings, and State
   breakdown snapshot. Retain ordinary workload limits, user workload decisions,
   cancellation, and incomplete-result handling. A repair may be verified even
   when later overlap/state analysis fails; report those statuses separately.

The independent verifier must enforce the rule, not merely trust a repairer's
`geometry_unchanged=True` flag. A coordinate checksum alone is insufficient: the
same vertices connected in a different order can describe a different shape.

## 4. User workflow

Recommended default: offer repair after an eligible invalid-file error, without
adding another permanent toggle to the input page.

1. Browse or drag-and-drop runs strict parsing as today.
2. For an eligible failure, show a concise recovery panel:
   **“This file has a formatting error that may be safely repairable.”**
   For a pure format mismatch use **“This file's contents do not match its extension.”**
   Helper: **“We’ll verify that its geometry is unchanged before analyzing it.”**
   Actions: **Repair & analyze**, **Choose another file**, and **Cancel**.
   Put the technical error/member/line details in an expandable disclosure.
3. One click starts repair, verification, and analysis. No second confirmation
   after verification. Show stages **Checking file**, **Repairing formatting**,
   **Verifying geometry**, then existing analysis progress.
4. On success, show one small result notice directly below the filename, above
   the Combined/state selector and result tabs:
   **“File formatting repaired. Geometry verified unchanged. Original preserved.”**
   Keep **Details** and **Save repaired copy…** with this notice. It must persist
   across scope/tab changes, including geography-unavailable results, without
   being copied into every state's rows or adding another mileage card.
5. Use the specific failure category below. Keep technical details available;
   never label an operational failure as evidence of changed geometry. Do not
   publish partial data as a completed repaired analysis.
6. Provide **Save repaired copy…** from the notice/details. Default to
   `<original stem>_repaired.kmz` (or `.kml`, matching the verified effective format);
   choose a collision-free path and never
   overwrite the source. Saving is optional; successful analysis does not depend
   on write access to Downloads or the input directory.

This makes repair and subsequent analysis automatic after one explicit action.
If a completely automatic retry on initial import is preferred later, use the
same verification gates and show the same result notice. That is a UX preference,
not a reason to broaden the technical repair rules.

Use the same workflow in modern and legacy GUI entrypoints, retries, and Adjust
Parameters. Keep the original filename visible. Reanalysis reuses the verified
snapshot during the session; reopening a changed original requires verification
again. Persist neither temporary repair paths nor stale repair approvals across
application restarts. Cancel and close must reject results from superseded jobs;
snapshot cleanup follows the ownership rules below rather than deleting the
verified source whenever an individual worker finishes.

### Recovery state machine and clean modal behavior

```text
Read and inspect frozen source
  ├─ valid input → existing analysis
  ├─ unsupported / unsafe / incomplete → specific recovery message
  └─ eligible defect → await Repair & analyze
                         ├─ Cancel → input page
                         └─ repair → verify → coverage check → existing analysis
```

- Use a typed `RepairRequired` preparation outcome carrying a source-session
  token and the captured parameter/options request. Do not derive eligibility
  from the displayed exception string or reread `current_file` on approval.
- Finish the preparation worker, retain its source session, and show one shared
  recovery panel. Repair & analyze creates a fresh job tied to that source token.
  Only one worker may analyze/repair the active source at once. Duplicate clicks
  and late callbacks cannot start or publish a second run.
- Freeze parameters and State breakdown while the decision panel is open.
  Restore control availability on every exit path. Browse/drop/retry behind the
  modal must not bypass the decision. Repair approval and workload approval are
  distinct; display them sequentially and never transfer one to the other.
- Initial Repair & analyze resumes the captured import settings. Later Retry
  analysis or Apply & Reanalyze captures the current settings and a new job ID
  while reusing only the verified source baseline. Cancelling Adjust Parameters
  changes neither the active source nor its analysis request.
- Reuse `ModalSurface`/`ModalBody` visuals: one blue primary action, neutral
  secondary actions, a short description, and collapsed technical details.
  Existing modal placement alone is not sufficient: implement keyboard focus
  containment/restoration, visible focus, Return/Space activation, and Escape
  cancellation. Use text as well as color to identify outcomes.
- Keep actions in a fixed footer; filenames, errors, and details wrap in a
  scrollable body. At 390×844 and 640×480, including 125–200% scaling, controls
  remain reachable without horizontal scrolling or clipped headers.
- **Choose another file** temporarily releases the repair panel for the native
  file picker. Cancelling the picker returns to the same eligible snapshot;
  accepting a replacement retires the old session after worker leases end.
- Cancellation means stopping work, not verification failure. After verified
  repair, failed/cancelled analysis retains a compact input-page notice and the
  verified snapshot for Retry analysis or Save repaired copy where eligible.
  Do not rerun the repair merely because overlap/state processing failed.
- If bytes are preserved but coverage validation fails, show **Formatting fixed;
  analysis blocked** with the actual missing/invalid data reason. Do not offer
  the normal verified-copy save action for an artifact that fails the complete
  saved-copy contract. A separate partial-salvage product is outside this plan.

| Outcome | User-facing wording / next action |
| --- | --- |
| Unsupported defect | “This formatting problem cannot be repaired automatically.” Choose another file; details explain the unsupported rule |
| Verification mismatch | “We could not verify that the geometry is unchanged.” Stop; keep the source untouched |
| Missing/invalid required data | “Formatting was repaired, but required geometry or linked files are unavailable.” Explain affected scope; block automatic analysis |
| Read/storage/resource error | “The file could not be read,” “Not enough space,” or the actual limit; retry where useful |
| Analysis fails after verified repair | “File repair verified. Analysis could not finish.” Retain accurate diagnostics and verified-source retry/save |
| Save fails | “The repaired copy could not be saved.” Results and verified input stay usable; offer Retry save |
| Cancel | Return to the owning input/result view without a red error or success claim |

Show diagnostic text as plain text, never rendered source HTML. Cap visible
detail size and hide private temporary paths. The success claim means this
application did not modify the source; it does not guarantee another program
has not edited the live file since the snapshot was captured.

### Actionable client feedback when safe repair is unavailable

Use one compact recovery panel with **Copy client request**, **Choose another
file**, and **Cancel** for confirmed source defects. Heading:
**“This file cannot be repaired safely.”** Helper:
**“We cannot verify a complete, unchanged system from this file.”**
Show a short issue summary and keep selectable, scrollable details behind the
existing disclosure. Keep actions visible at small window sizes/high DPI, support
keyboard focus, and wrap long member names. No new input-page toggle or extra
modal for each defect. Clipboard failure leaves the request selectable and shows
a brief copy-failed message. Copying never sends anything to the client.

Generate the request from structured findings, using this template:

> Please ask the client to provide a new complete KML/KMZ that fixes the following issues:
>
> - [Original member and known location]: [observed issue]. [Specific corrective action].
>
> Please re-export from the original GIS data while preserving all pipeline vertices,
> separate paths, and intended centerlines. Do not delete features or vertices merely
> to make the file load.
>
> [If applicable: Validation stopped at the first XML error; additional issues may remain.]

Include only confirmed observations and applicable instructions. XML failure does
not prove a specific shape is missing, and preservation of supplied bytes cannot
establish whether the sender exported their entire client system in the first
place. Never label the request an exhaustive diagnosis unless the relevant graph
was actually inspected; record uninspected documents and truncated findings.

| Finding category | What to explain and request |
| --- | --- |
| Unclosed/mismatched tag, trailing content, multiple roots | Give the observed XML error and location. Request a complete valid re-export; do not claim which geometry was lost or truncate to the parseable part |
| Invalid/non-finite/out-of-range coordinate, malformed track, independently verified short path | Identify the original feature/path and offending tuple position if known. Ask the data owner to correct that source record and re-export; do not propose values, axis swaps, vertex deletion, or automatic connection of remaining points |
| Missing/ambiguous geometry namespace or incorrectly cased reserved element | Identify the token/binding. Ask for correct KML namespaces and element spelling from the source exporter; do not guess a URI or silently relabel geometry |
| Encoding conflict, illegal character, ambiguous entity, broken description markup | Report the observed mismatch/text location. Request consistently encoded valid XML and properly escaped metadata without changing source geometry; avoid claiming an encoding was detected when it was not |
| Missing/out-of-base/remote required NetworkLink | Name the target and explain it was not available to this offline analysis. Request a complete self-contained KMZ including the linked centerlines; do not call an otherwise valid remote link corrupt |
| Damaged ZIP/CRC, duplicate or ambiguous member paths, absent primary KML | Request a freshly exported/resupplied complete KMZ; do not salvage intact members and treat them as the full system |
| Encrypted archive or supported-policy limitation such as DTD | Explain the unsupported packaging/construct; ask for an unencrypted/self-contained standard export as applicable, not deletion of shapes |
| No supported centerlines / unsupported geometry representation | Explain which supported centerline representation is needed; do not turn polygon outlines, models, or point sequences into pipeline mileage |

Keep verification and operational outcomes separate from this client-defect flow:

- A failed independent proof means **“We could not verify this repair”**, not
  proof that the original geometry was altered or incorrect. Keep technical
  details for app support; do not invent a client correction. Offer another file
  and a copyable diagnostic report, not a false list of source errors.
- File permissions, disk space, source changes during acquisition, cancellation,
  and application work/size limits get specific local actions. A limit is not
  source corruption. Do not suggest dropping geometry or splitting the client's
  system into independent analyses to bypass it; splitting can change overlaps.
- Missing linked content is a source completeness issue; a locally inaccessible
  linked file may instead be a permission issue. Preserve the actual I/O cause.
- If repairable formatting and unrepairable data coexist, list the confirmed
  blockers and do not auto-analyze the repaired subset or offer a verified copy.

Each finding needs a stable issue code, category, original member, source byte
offset and line/column when known, original feature ordinal/ID and geometry/path
ordinal when known, observation, recommended action, and inspection coverage.
Translate candidate offsets through the patch map; label coordinates/locations
with their indexing convention. Escape all source text for plain display/export.
Do not rely on pipeline names alone because names and IDs can repeat.

The current parser often reports only feature names. Add source-location/path
identity to preparation diagnostics before promising precise locations in the UI.
Also group causal duplicates: one invalid coordinate currently yields
`invalid_coordinate` followed by `short_linestring`; tracks can yield
`invalid_gx_coord` followed by `short_gx_track`. Report the original coordinate
defect once with related diagnostic codes. Do not assert that the original line
had too few vertices merely because the parser rejected its coordinate list.
[Current parser](src/pipeline_calculator/parsers/kml_kmz.py).

## 5. Backend and integration design

### Proposed modules and interfaces

- `parsers/repair.py`: bounded detection, lexical scanning, the four versioned
  catalog rules, patch composition, and independent verification. Pure
  data operations; no GUI, geodesic processing, or network access.
- `parsers/source.py` or a small equivalent abstraction: effective document bytes
  plus stable original source names. Keep normal file/KMZ selection and local
  link resolution rules centralized in the existing parser.
- Immutable `RepairPolicy`, `RepairReport`, and structured client-finding records;
  explicit statuses such as
  `not_needed`, `eligible`, `verified`, `unsupported`, `verification_failed`, and
  cancellation as existing control flow. Preserve JSON-native serialization.
- App-owned `SourceSession`, separate from the serializable report, containing
  immutable source/document snapshots, manifest, original display identity,
  selected primary/link graph, and verified repair overlay. Background work
  receives a lease; transient file handles and parsed objects never enter JSON.
- Typed invalid-input/repair-eligibility information instead of GUI code matching
  fragments of English exception strings. Preserve exception causes through the
  parser and analyzer wrappers.

Existing integration points:

| Area | Current code and proposed change |
| --- | --- |
| Strict parsing | [`kml_kmz.py`](src/pipeline_calculator/parsers/kml_kmz.py): `_parse_kml_bytes`, `_parse_kmz`, `_parse_kml_file`; expose structured failure/member information and accept verified document overrides |
| Analyzer | [`analyzer.py`](src/pipeline_calculator/core/analyzer.py): `analyze_complete` currently parses then calls reusable `analyze_features`; share this orchestration so the verified parse result is not parsed again through a different path |
| Background execution | [`analysis_controller.py`](src/pipeline_calculator/gui/controllers/analysis_controller.py): snapshot repair policy/source token alongside analysis parameters/options and keep one active worker |
| Progress/cancellation | [`analysis_session.py`](src/pipeline_calculator/gui/controllers/analysis_session.py), core execution/progress: reserve progress for repair stages; do not reach 100% before analysis completes |
| Both GUIs | [`main_window.py`](src/pipeline_calculator/gui/main_window.py), [`pipeline_calculator_v3.py`](src/pipeline_calculator_v3.py): replace eligible generic error dialogs with the shared recovery panel |
| Provenance/export | Existing JSON, [`xlsx.py`](src/pipeline_calculator/export/xlsx.py), [`geography_xlsx.py`](src/pipeline_calculator/export/geography_xlsx.py), and [`package.py`](src/pipeline_calculator/export/package.py): include repair provenance without changing mileage tables |

Keep original archive member names and internal source identities stable.
Temporary file paths must not become feature IDs, display names, or retry targets.
Preserve disconnected coordinate paths and duplicate source names exactly as the
existing parser represents them.

Forward the full request independently of `state_breakdown`. Several current
entrypoint/session/controller calls forward `options` only when that flag is
true; adding a repair flag to that object without fixing those gates would
silently disable repair in ordinary mode. Prefer a separate source/repair request
field and explicit public API policy. Library callers default to strict mode and
may opt into verified repair without opening GUI dialogs.

Prepare once per source session; start each analysis with fresh, analysis-owned
feature dictionaries copied from an immutable normalized baseline. The analyzer
mutates pipeline data (including generated segment caches), so a previously
analyzed `ParseResult` cannot serve as preservation evidence or a reusable mutable
baseline. Reanalysis reuses verification, never stale overlap/state results.

Expose separate report fields for repair verification, parser coverage,
combined-analysis completion, geography completion, and saved-copy validation.
Use one cancellation context through all worker stages. Allocate measurable
preparation/repair progress ahead of normal analysis and map ordinary and state
progress into the remaining interval; existing hard-coded progress ranges cannot
simply be reused as independent 0–100% passes. Indeterminate progress is acceptable
while total work is unknown. Never report completion when awaiting repair choice.

### KMZ and linked documents

- Follow the existing primary-document selection and local link graph; do not
  analyze all KML members merely because a repair scanned the archive.
  **Pin the original primary member before edits.** Without `doc.kml`, the parser
  falls back to the largest uncompressed KML. A repaired child may become larger
  than its parent. Session analysis must retain the original primary; saved-copy
  validation must independently reopen through ordinary selection and match that
  primary, full reachable graph/order, source identities, and geometry. If the
  saved candidate selects differently, decline saving it and explain why. Do not
  rename members, add padding, inject links, or discard members to force selection.
  Equal-size ties and nested `doc.kml` ordering also need regression coverage.
- Detect eligible failures in reachable linked KML as well as the root document.
  The current parser reports linked parse failures as diagnostics instead of
  raising a fatal root error; repair must handle that path explicitly before
  expensive analysis. Batch eligible member edits into one verified job.
- Unreachable members remain untouched and retain their existing diagnostics.
  Report precisely which documents were repaired/validated; do not claim to have
  validated the entire archive if some documents are outside the analysis graph.
- Keep ZIP entry names/order, archive comments, applicable entry metadata, and
  every untouched member's uncompressed bytes. Compression bytes may differ in a
  saved copy. Define an explicit supported metadata policy: retain timestamps,
  entry comments, platform/permission attributes and supported compression;
  regenerate CRC/size/offset/ZIP64 fields. Do not blindly replay stale structural
  extra fields. Decline unsupported metadata whose preservation matters rather
  than claim an exact archive clone. Verify CRCs/member hashes and reopen the
  finished copy before success.
- Never extract archive paths directly to the filesystem. Validate **all** members
  when preparing a portable saved copy, including resources and directory entries.
  Reject exact/normalized/case-portability collisions, traversal, drive/UNC/absolute
  paths, special/symlink entries, encrypted or unsupported compressed members, and
  inconsistent member metadata. Preserve accepted names rather than sanitizing
  names and silently breaking links.
  For v1, perform this archive/member preflight and bounded CRC/hash streaming
  **before offering repair**, including unused resources. Corruption in an unused
  member therefore blocks repair of this archive, not merely saving. Revalidate
  the staged saved copy separately. Unreachable KML's payload/CRC is checked but
  its XML/geometry is not analyzed; the report must retain that scope distinction.
- Standalone KML with local linked documents uses an in-memory document overlay
  to keep relative resolution correct. Initially, **Save repaired copy** supports
  self-contained KML and KMZ that pass independent saved-copy validation. Define
  **self-contained** beyond NetworkLinks: inspect Icon/Model resources, external
  `styleUrl`, `xml:base`, self-filename references, and other base-sensitive local
  references. Changing directory or basename must not break dependencies. For
  standalone datasets with such references, disable save with an explanation
  until dependency packaging is implemented; verified session analysis remains
  supported. Preserve remote visual resource references without fetching them;
  never claim that rendered external assets were verified. Unrecognized reference
  semantics block portability claims. Apply the same check to external local
  references escaping a KMZ package.
- Reuse current limits: 64 MiB/document, 256 MiB aggregate KML, 1,024 documents,
  10,000 archive entries. Add a 256 MiB aggregate uncompressed repair-package
  budget covering resources too, cancellation checkpoints, bounded diagnostics,
  and one candidate transaction for the entire source snapshot. Larger packages receive an explicit
  unsupported-repair message. Bound the source archive snapshot itself (256 MiB
  compressed), enforce cumulative actual decompressed bytes rather than trusting
  `ZipInfo.file_size`, and stream resource copies/hashes with cancellation. Never
  use unbounded `ZipFile.testzip()` in the production path; the investigation used
  it only for the tiny hash-pinned sample.

Run safety preflight **before** initial XML-tree construction, not after an error
has already expanded entities or exhausted memory. Reject actual DTD declarations
using the structural reader's declaration callbacks; comment/CDATA lookalikes
are not declarations. Initial technical ceilings for the repair-capable reader:
256 element nesting levels, 500,000 elements per source snapshot, 256 attributes
per element, and 10,000 diagnostic records with a final truncation notice. Stream
coordinate/text inspection within the existing byte limit; avoid duplicating
the entire document into token lists. Use bounded chunks/checkpoints for hashes,
scanning, decompression, and copies. Benchmark these constants against existing
fixtures before release and document any adjustment. Supported ordinary files
retain their calculations; rejecting DTD/over-complex XML earlier is an explicit
input-policy change to cover with compatibility tests, not silently call unchanged.

### Results and storage

Add an optional versioned `input_repair` result object containing original source
name/hash, stable source-manifest identity, primary/link graph identities,
effective-document hashes, rule/verifier/app versions, repaired member names,
supplied extension/effective format, edit counts, verification/coverage status/counts,
structured findings/inspection limits, and original-preserved status. Keep exact
patch details in the expandable report; avoid duplicating geometry in this object.

Ordinary workbooks currently have no metadata sheet. Add **Analysis Details**
only for repaired ordinary results; append the same shared repair rows to the
existing geography Analysis Details sheet. Preserve the existing worksheet
contracts for unrepaired inputs. Keep source text literal, retain formula escaping,
and summarize long edit lists with counts rather than silently truncate them.
JSON carries the structured report. Validate its schema and JSON-native types
before either export route: ordinary JSON's existing `default=str` must not hide
bytes, paths, dataclasses, or handles in this new object. Represent patch records
as removed/inserted UTF-8 text plus original byte offset/length, not raw bytes.
Format-only correction has a rule receipt and zero patches. Do not include absolute user
paths or private temporary filenames in portable provenance.
Do not emit extra files into every state folder. A separately saved repaired
source is an import artifact, distinct from analyzed/partitioned output KMZs.
Do not label state KMZs as byte-preserved repaired originals; they are transformed
analysis outputs derived from the verified input. Keep repair provenance attached
to the complete result snapshot through export preparation and all state views.

### Snapshot lifetime and saved-copy transaction

- `SourceSession` outlives an individual `AnalysisSession`. Keep verified source
  bytes for the active results/retry workflow, even when later analysis fails or
  is cancelled. Remove disposable job scratch files on completion/failure, not
  the retained verified source. Cancel before verification discards candidates.
- Replace/close retires the source only after its analysis/save/export leases are
  released. Never delete temporary input underneath a worker. Returning to the
  file picker alone does not abandon the old source; accepting new input does.
  Do not persist repair eligibility/approval across application restarts.
- Crash cleanup is restricted to application-owned marked temporary directories,
  without following links or deleting live-session data. Plain result exports
  may still work if retained input expires, but Save repaired copy must then say
  **Verified source no longer available** rather than reread a possibly changed file.
- Save runs on a worker from the retained snapshot. Its transaction is:
  **reserve destination → stage copy → verify manifest and ordinary reimport →
  publish without replacement**. Reject source-path aliases, case variants,
  symlinks/hardlinks to the original, and existing destinations. Recheck at
  publication; a collision causes a new name or clean retry, never overwrite.
- File creation uses an atomic no-clobber publication method supported on the
  destination filesystem. If that guarantee is unavailable, fail with a clear
  message instead of falling back to replacement. Keep the verified effective
  file type; only rule D changes an incorrect filename suffix. Pure routing
  correction copies original bytes exactly rather than recompressing the ZIP.
- Preserve member order/names, untouched member payload hashes, inverse patches,
  supported metadata, selected document graph, full feature identity/path
  projection, and unrounded mileage. Reopening the staged file through the
  ordinary parser is required before publication. A passing session run does
  not alone certify a portable saved archive.
- Standalone KML currently records `source_kml` as an absolute path. Use canonical
  document identity and one explicit original-root → staged/saved-root relocation
  mapping for verification; a new basename is not a geometry change. Preserve
  every Placemark ID, OBJECTID, feature ordinal/name, and path identity. Do not
  broadly strip source fields to make comparisons pass. KMZ member names must
  still match exactly; standalone dependency relocation remains unsupported.
- Keep source-content identity and saved-archive SHA-256 separate: recompression
  changes the archive bytes. Bind the save receipt to the verified manifest.
  A successful save does not change the analysis filename, source session,
  results, State breakdown selection, or parameters.
- Disk-full/permission/cancellation/verification failure removes staging output
  and leaves the verified snapshot/results usable. Cancelling Save As is a no-op.
  If cancellation arrives after atomic publication, report the completed save
  accurately; do not silently delete a finished artifact. Existing
  `run_background_action()` provides responsiveness but not these transaction or
  cancellation guarantees by itself.

## 6. Verification and acceptance

### Core fixtures

- Minimal broken `xsi:schemaLocation` example and a manually authored valid
  counterpart: exact normalized features, IDs, path boundaries, and source lengths
  must agree. The valid counterpart must not be produced by the repair function
  under test.
- Each added catalog rule: authored valid counterparts, eligible variants, and
  near-miss refusals in both supported containers where applicable. Include the
  complete multi-rule transaction; compare exact source geometry and full parser
  projection, not just total mileage. The investigation script uses fixed known
  fixture offsets and is not a substitute for testing the production detector.
- Supplied August 2023 sample, hash-pinned: reproduce the original line-6 error;
  one 54-byte insertion; 42 pipelines / 44 paths / 2,139 vertices; unchanged raw
  coordinate blocks and inverse-patch identity; mileage matches the report within
  `max(0.001 m, total_meters * 1e-10)`. Preserve the source. Keep the customer file
  local/private; use synthetic fixtures for committed regression tests unless
  this dataset is explicitly approved for inclusion.
- MultiGeometry, repeated IDs/names, altitude values, zero-length/repeated
  vertices, polygon holes, points, tracks/timestamps, antimeridian lines, and
  disconnected paths: no geometry repair, regrouping, or silent removal.
- Real attributes versus lookalikes in comments/CDATA/descriptions; quoted `>`;
  Unicode, BOM, CRLF, differing attribute order/quotes, nested declarations,
  conflicting namespace bindings, missing prefixes on geometry, `xsi:type`/`nil`.
- Broken tags, truncation, DTD/custom entities, invalid UTF-8, excessive input,
  and damaged ZIPs must stop without verified-repair success. Legitimate predefined
  and numeric references must pass when otherwise eligible. Invalid coordinates
  may coexist with a proven metadata insertion: assert `byte_preservation=passed`,
  `coverage=failed`, and `automatic_analysis=blocked`, never one misleading
  all-purpose “verified” flag.

### Integration acceptance

- Accepted files within the documented reader policy bypass repair and preserve
  existing result/export behavior, including ordinary partial-analysis diagnostics.
- Known-good versus repaired synthetic systems produce equivalent original,
  overlap, and state results using identical settings; include cross-border and
  shared-border cases. No repair rule is allowed to fix a geography reconciliation
  error by altering the source.
- Root and linked-member failures, nested KMZ paths, link cycles, missing links,
  unreachable members, assets, duplicate archive paths, and local KML references.
- Inject a verifier failure or unauthorized edit to coordinates, path grouping,
  IDs, or links: assert the analyzer is never called.
- Browse/drop/retry/reanalysis in both GUIs; State breakdown ON/OFF; source changed
  during the workflow; read-only source directory; cancelled/superseded jobs;
  workload confirmation; no freeze or late-result publication.
- Reopen saved repaired copy through the normal strict parser and verify identical
  extracted geometry and original mileage. Assert original files and untouched
  resource bytes are unchanged, no overwrite occurs, and reports survive export.
- Offline packaged Windows and macOS smoke checks, clean temporary storage, and
  readable recovery notices/dialogs at small sizes and high DPI.

### Audit regressions required before release

| ID | Scenario | Required outcome |
| --- | --- | --- |
| R-01 | Parent 515 bytes; reachable child 505 bytes; child becomes 559 bytes after repair | Session keeps the original parent and both pipelines; ordinary saved-copy selection mismatch blocks save |
| R-02 | Linked KML contains another linked KML, several members need repair, cycles/ties/nested `doc.kml` exist | Bounded traversal and document order remain stable; no child-only or duplicate analysis |
| R-03 | Mixed LineString, track, point, polygon, altitude, repeated names/IDs in one source | Exact source preservation plus explicitly correct normalized extraction; no false mismatch from line-before-track ordering |
| R-04 | Original/dependency changes during capture or is deleted/changed while awaiting Repair | Unstable capture is declined; accepted frozen input is used consistently without reopening live paths |
| R-05 | Analysis mutates its pipeline/segment dictionaries, then parameters are changed | Fresh analysis-owned inputs; verified baseline and previous results remain unaffected |
| R-06 | State breakdown OFF, ON, and switched during a later parameter draft | Repair request reaches every layer independently; Apply uses the new options and retained source, Cancel changes neither |
| R-07 | Double Repair click, cancelled file picker, Escape, root drop behind modal, late callback | One active request; captured settings retained; focus restored; no bypass or unexpected run |
| R-08 | Repair verifies but overlap/state analysis fails or is cancelled | Accurate separate statuses; retained source enables safe retry/save; no repeat repair or lost evidence |
| R-09 | Switch Combined/state/tabs, including failed scope rendering | One persistent repair notice; full provenance remains available to export |
| R-10 | Save moves standalone KML containing local icons/styles/models/self references | Disable unsupported portability; never silently break relative references |
| R-11 | Save collision/race, source hardlink/symlink alias, disk-full, permission error, close/cancel | No overwrite or partially published artifact; staging removed when safe; correct worker/source ownership |
| R-12 | Reopen repaired copy a second time and inspect again | Strict parse with the same primary/graph/feature identity; repair is `not_needed`; no repeated header insertions |
| R-13 | Deep XML, huge attributes/token volume, forged ZIP sizes, oversized resource, malformed UTF-8/DTD | Stop within defined resource bounds before unbounded allocation; no XML/entity/network fallback |
| R-14 | JSON ordinary vs geography; XLSX text begins with `=`; long report values | Equivalent structured report, escaped source text, explicit summaries, no handles/bytes/path leakage |
| R-15 | Repeated open/repair/reanalyze/save/replace/close; cancel during each stage | No orphan workers, source leases, Tk callbacks, temporary data leaks, or regressions to unrepaired files |
| R-16 | XML S before declaration, optional single BOM, CRLF, multibyte metadata | Only permitted prefix bytes removed; original error locations map correctly; geometry and declaration unchanged |
| R-17 | Literal ampersands in each eligible leaf; existing references; end-of-leaf ampersand | Exactly permitted ampersands encoded; intended decoded text, geometry, identity, and unrelated bytes preserved |
| R-18 | Similar text in foreign namespace, attribute, coordinate, href, ExtendedData, CDATA/comment, raw HTML; `A&B`, undefined or unfinished entities | No false eligible match, double escaping, guessing, sanitization, or metadata-derived geometry; valid lookalikes remain untouched |
| R-19 | Wrong top-level suffix in both directions; assets/multiple members; polyglot, arbitrary ZIP, misnamed dependency, base-sensitive plaintext source | Only verified unambiguous top-level routing accepted; unchanged payload, pinned graph, correct saved suffix and ordinary reimport; unsafe cases refused |
| R-20 | Several allowlisted defects per document and linked graph; routing plus XML edits; one unrepairable blocker; limits/cancellation mid-scan | One bounded deterministic transaction and one prompt; all proofs pass or no automatic analysis; no partially repaired success |
| R-21 | Single BOM, valid UTF-16, no declaration, trailing XML whitespace, valid references/CDATA/comments | Ordinary accepted files unchanged; no unnecessary repair prompt, trimming, transcoding, or export contract change |
| R-22 | Duplicate names/IDs, original multibyte offsets, invalid coordinate plus cascading short-line diagnostic, first-error stoppage, diagnostic cap | Correct original member/path or explicit unknown location; one causal client issue; honest inspection coverage; no fabricated coordinate or missing-shape claim |
| R-23 | Client-request copy/selectable fallback, keyboard/high DPI, technical detail wrapping; source error vs permissions/limits/proof failure | Actionable source-only correction request; operational/support remedies separate; no request to delete geometry or split overlap analyses |
| R-24 | Corrupt proposed patch targeting coordinates, attributes, grouping, or links; patch overlap/reordering; verifier trusts supplied rule labels | Independent verification rejects unauthorized location/content/context and blocks analyzer invocation; all reverse patches restore exact bytes |

R-01 is an observed current-parser behavior, not a hypothetical risk. The
[audit evidence](docs/validation/kml-repair-plan-audit.json) records the member sizes,
selected primary, returned pipeline names, diagnostic codes, and parser fingerprint.
The [synthetic reproducer](scripts/validation/inspect_repair_primary_selection.py)
requires no customer data. It demonstrates why coordinate hashes and unchanged
mileage alone cannot certify that the same dataset was selected: both synthetic
paths even have the same coordinates.

Use existing parser/execution/state/export tests plus focused repair suites.
Native GUI tests must use the repository's isolated desktop runner in
`tests/conftest.py` / `scripts/validation/gui_process.py`; do not interrupt the
user's desktop. Assert bounded response times, stage progress without backward
jumps, cleanup after destruction, and availability of fixed-footer actions.

## 7. Audit findings resolved in this plan

| Priority | Gap identified during audit | Required resolution |
| --- | --- | --- |
| P1 | Repaired member size can change primary-document selection | Pin primary and traversal; verify ordinary saved-copy reimport, refuse changed selection |
| P1 | “Cleanup on success” contradicts later retry/save | App-owned source session with worker leases and explicit lifetime |
| P1 | Verification compared raw order/geometry against lossy normalized records | Distinct byte/structure proof and documented supported-geometry projection |
| P1 | Namespace fallback and ZIP limits were not a complete preflight policy | Bounded structural inspection before XML tree construction; actual-byte/resource checks |
| P1 | Repair option could disappear when State breakdown is OFF | Full request forwarding independent of state mode; typed preparation outcomes |
| P2 | Repair success conflated with later analysis/coverage/save status | Separate statuses and recovery paths, including verified-but-analysis-failed |
| P2 | Recovery modal lacked focus, keyboard, cancellation, and ownership contracts | Shared accessible panel and explicit lifecycle/state machine |
| P2 | “Self-contained KML” ignored non-NetworkLink resources | Conservative base-reference inventory and portable-save eligibility |
| P2 | Ordinary workbook metadata destination and JSON types were unspecified | Shared Analysis Details writer and validated JSON-native provenance |
| P1 | A single-format fix can mask other independently repairable defects | Explicit four-rule scope, one bounded composed candidate, strict whole-source verification |
| P1 | Successful XML parsing can still silently ignore incorrectly cased geometry | Protected source inventory and app-projection coverage; report ambiguity, never auto-correct case |
| P2 | Generic failure text leaves users guessing what to ask a client | Structured original-location findings, causal grouping, copyable client request, honest inspection limits |
| P2 | “Repair” could be presented for already valid BOM/encoding/CDATA cases | Already-valid controls and no-op bypass; no encoding guessing |

## 8. Delivery order and readiness

1. Implement synthetic fixtures, typed errors, bounded structural reader, all
   four catalog rules, exact edit/routing records, and independent verification.
   Prove the supplied defect, combined cases, and near-miss refusals; establish
   reproducible parser-version expectations.
2. Integrate owned source snapshots/overlays and parse-once preparation with fresh
   analysis inputs, cancellation, stable primary/feature identities, complete
   option forwarding, and linked-document diagnostics.
3. Add the shared recovery panel, copyable client requests, session reuse,
   provenance, and optional saved copy. Review success/refusal/operational wording.
4. Complete R-01 through R-24 and parser, analysis, geography, UI, and export
   regressions; review sample recovery/success/error screens and saved artifacts.
   Build packaged applications only after the feature is implemented and verified.

**Feasibility is established for the supplied defect and synthetic examples of
the catalog additions.** No product decision requires user input before this
defined implementation scope. The recommended UX is one-click Repair & analyze
after an eligible failure; always-automatic retry is a separate future preference.
Repairs outside this catalog, standalone linked-file save packaging, and full KML
XSD conformance remain excluded. The production scanner, independent verifier,
integration regressions, and packaged checks are implementation/release gates;
mechanism probes do not prove those gates have passed. Future unknown errors get
truthful diagnosis and a client/support next action, never a guessed repair.
