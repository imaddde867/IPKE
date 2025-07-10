# Database Schema

This document outlines the PostgreSQL database schema used by the Industrial Knowledge Extraction System. The schema is managed using SQLAlchemy ORM.

## Models and Tables

### `Document` Table

Stores metadata and extracted high-level information for each processed document.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `filename`            | `String`  | Original filename                               | `INDEX`          |
| `file_type`           | `String`  | Type of file (e.g., `document`, `image`, `video`) |                  |
| `extracted_text`      | `Text`    | Full text extracted from the document           |                  |
| `metadata_json`       | `JSON`    | Raw metadata from Tika or other processors      |                  |
| `classification_category` | `String`  | Predicted document category                     |                  |
| `classification_score`| `Float`   | Confidence score of the classification          |                  |
| `status`              | `String`  | Processing status (e.g., `processed`, `failed`) |                  |
| `processing_timestamp`| `DateTime`| Timestamp of when the document was processed    | `DEFAULT CURRENT_TIMESTAMP` |
| `document_sections`   | `JSON`    | Extracted sections of the document (key-value pairs) |                  |

**Relationships:**
- One-to-many with `ExtractedEntity` (via `document_id`)
- One-to-many with `KeyPhrase` (via `document_id`)
- One-to-many with `Equipment` (via `document_id`)
- One-to-many with `Procedure` (via `document_id`)
- One-to-many with `SafetyInformation` (via `document_id`)
- One-to-many with `TechnicalSpecification` (via `document_id`)
- One-to-many with `Personnel` (via `document_id`)

### `ExtractedEntity` Table

Stores Named Entities extracted from the document text.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `document_id`         | `Integer` | Foreign key to `Document` table                 | `FOREIGN KEY (documents.id)` |
| `text`                | `String`  | The extracted entity text                       |                  |
| `entity_type`         | `String`  | Type of entity (e.g., `PER`, `ORG`, `LOC`, `EQUIPMENT`) |                  |
| `score`               | `Float`   | Confidence score of the entity extraction       |                  |
| `start_char`          | `Integer` | Starting character index in `extracted_text`    |                  |
| `end_char`            | `Integer` | Ending character index in `extracted_text`      |                  |

**Relationships:**
- Many-to-one with `Document`

### `KeyPhrase` Table

Stores key phrases extracted from the document text.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `document_id`         | `Integer` | Foreign key to `Document` table                 | `FOREIGN KEY (documents.id)` |
| `phrase`              | `String`  | The extracted key phrase                        |                  |

**Relationships:**
- Many-to-one with `Document`

### `Equipment` Table

Stores structured data about equipment mentioned in documents.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `document_id`         | `Integer` | Foreign key to `Document` table                 | `FOREIGN KEY (documents.id)` |
| `name`                | `String`  | Name of the equipment                           | `INDEX`          |
| `type`                | `String`  | Type of equipment (e.g., `Pump`, `Motor`, `Valve`) |                  |
| `specifications`      | `JSON`    | Key-value pairs of equipment specifications     |                  |
| `location`            | `String`  | Location of the equipment (if extracted)        | `NULLABLE`       |
| `confidence`          | `Float`   | Confidence score of the extraction              | `NULLABLE`       |

**Relationships:**
- Many-to-one with `Document`

### `Procedure` Table

Stores structured data about procedures mentioned in documents.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `document_id`         | `Integer` | Foreign key to `Document` table                 | `FOREIGN KEY (documents.id)` |
| `title`               | `String`  | Title of the procedure                          | `INDEX`          |
| `steps`               | `JSON`    | Ordered list of procedure steps                 |                  |
| `category`            | `String`  | Category of the procedure (e.g., `Startup`, `Maintenance`) | `NULLABLE`       |
| `confidence`          | `Float`   | Confidence score of the extraction              | `NULLABLE`       |

**Relationships:**
- Many-to-one with `Document`

### `SafetyInformation` Table

Stores structured data about safety information.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `document_id`         | `Integer` | Foreign key to `Document` table                 | `FOREIGN KEY (documents.id)` |
| `hazard`              | `String`  | Description of the hazard                       |                  |
| `precaution`          | `String`  | Recommended precaution or action               |                  |
| `ppe_required`        | `String`  | Personal Protective Equipment required          |                  |
| `severity`            | `String`  | Severity of the hazard (e.g., `High`, `Medium`) | `NULLABLE`       |
| `confidence`          | `Float`   | Confidence score of the extraction              | `NULLABLE`       |

**Relationships:**
- Many-to-one with `Document`

### `TechnicalSpecification` Table

Stores structured data about technical specifications.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `document_id`         | `Integer` | Foreign key to `Document` table                 | `FOREIGN KEY (documents.id)` |
| `parameter`           | `String`  | Name of the technical parameter                 |                  |
| `value`               | `String`  | Value of the parameter                          |                  |
| `unit`                | `String`  | Unit of measurement (e.g., `C`, `psi`, `mm`)    | `NULLABLE`       |
| `tolerance`           | `String`  | Tolerance of the measurement (e.g., `+/- 0.1mm`) | `NULLABLE`       |
| `confidence`          | `Float`   | Confidence score of the extraction              | `NULLABLE`       |

**Relationships:**
- Many-to-one with `Document`

### `Personnel` Table

Stores structured data about personnel mentioned in documents.

| Column Name           | Type      | Description                                     | Constraints      |
|-----------------------|-----------|-------------------------------------------------|------------------|
| `id`                  | `Integer` | Primary key, unique identifier                  | `PRIMARY KEY`, `INDEX` |
| `document_id`         | `Integer` | Foreign key to `Document` table                 | `FOREIGN KEY (documents.id)` |
| `name`                | `String`  | Name of the personnel                           | `INDEX`          |
| `role`                | `String`  | Role or job title                               |                  |
| `responsibilities`    | `Text`    | Description of responsibilities                 | `NULLABLE`       |
| `certifications`      | `JSON`    | List of certifications (e.g., `["OSHA 30"]`)  | `NULLABLE`       |
| `confidence`          | `Float`   | Confidence score of the extraction              | `NULLABLE`       |

**Relationships:**
- Many-to-one with `Document`
