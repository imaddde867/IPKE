import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState('');
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState(null);
  const [documentDetails, setDocumentDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch all documents on component mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Fetch details for selected document
  useEffect(() => {
    if (selectedDocumentId) {
      fetchDocumentDetails(selectedDocumentId);
    }
  }, [selectedDocumentId]);

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('http://localhost:8000/documents/');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      // Handle paginated response - extract items array
      const documentsArray = data.items || data || [];
      setDocuments(Array.isArray(documentsArray) ? documentsArray : []);
    } catch (error) {
      console.error("Error fetching documents:", error);
      setError(`Failed to fetch documents: ${error.message}`);
      setDocuments([]); // Ensure documents is always an array
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setUploadMessage('Please select a file first!');
      return;
    }

    setUploadMessage('Uploading and processing...');
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/uploadfile/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setUploadMessage(`Upload successful: ${data.info}. Processing in background.`);
      setSelectedFile(null); // Clear selected file
      // Give some time for processing before refreshing list
      setTimeout(fetchDocuments, 3000); 
    } catch (error) {
      console.error("Error uploading file:", error);
      setUploadMessage(`Upload failed: ${error.message}`);
    }
  };

  const handleSearch = async () => {
    if (!searchTerm) {
      setSearchResults([]);
      return;
    }
    try {
      const response = await fetch(`http://localhost:8000/search/?query=${searchTerm}&field=extracted_text`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setSearchResults(data);
    } catch (error) {
      console.error("Error searching documents:", error);
    }
  };

  const fetchDocumentDetails = async (docId) => {
    try {
      const [docRes, entitiesRes, keyphrasesRes, equipmentRes, proceduresRes, safetyRes, techSpecsRes, personnelRes, sectionsRes] = await Promise.all([
        fetch(`http://localhost:8000/documents/${docId}`),
        fetch(`http://localhost:8000/documents/${docId}/entities/`),
        fetch(`http://localhost:8000/documents/${docId}/keyphrases/`),
        fetch(`http://localhost:8000/documents/${docId}/equipment/`),
        fetch(`http://localhost:8000/documents/${docId}/procedures/`),
        fetch(`http://localhost:8000/documents/${docId}/safety_info/`),
        fetch(`http://localhost:8000/documents/${docId}/technical_specs/`),
        fetch(`http://localhost:8000/documents/${docId}/personnel/`),
        fetch(`http://localhost:8000/documents/${docId}/sections/`),
      ]);

      const docData = await docRes.json();
      const entitiesData = entitiesRes.ok ? await entitiesRes.json() : [];
      const keyphrasesData = keyphrasesRes.ok ? await keyphrasesRes.json() : [];
      const equipmentData = equipmentRes.ok ? await equipmentRes.json() : [];
      const proceduresData = proceduresRes.ok ? await proceduresRes.json() : [];
      const safetyData = safetyRes.ok ? await safetyRes.json() : [];
      const techSpecsData = techSpecsRes.ok ? await techSpecsRes.json() : [];
      const personnelData = personnelRes.ok ? await personnelRes.json() : [];
      const sectionsData = sectionsRes.ok ? await sectionsRes.json() : {};

      setDocumentDetails({
        ...docData,
        entities: entitiesData,
        key_phrases: keyphrasesData,
        equipment: equipmentData,
        procedures: proceduresData,
        safety_info: safetyData,
        technical_specs: techSpecsData,
        personnel: personnelData,
        document_sections: sectionsData,
      });
    } catch (error) {
      console.error("Error fetching document details:", error);
      setDocumentDetails(null);
    }
  };

  const handleDocumentClick = (docId) => {
    setSelectedDocumentId(docId);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Industrial Knowledge Extraction System</h1>
      </header>
      <main>
        <section className="upload-section">
          <h2>Upload Document</h2>
          <input type="file" onChange={handleFileChange} />
          <button onClick={handleFileUpload}>Upload</button>
          {uploadMessage && <p>{uploadMessage}</p>}
        </section>

        <section className="search-section">
          <h2>Search Documents</h2>
          <input
            type="text"
            placeholder="Enter search term"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          <button onClick={handleSearch}>Search</button>
          <div className="search-results">
            {searchResults.length > 0 ? (
              <ul>
                {searchResults.map((doc) => (
                  <li key={doc.document_id} onClick={() => handleDocumentClick(doc.document_id)}>
                    {doc.filename} (ID: {doc.document_id}) - {doc.classification_category}
                  </li>
                ))}
              </ul>
            ) : (
              searchTerm && <p>No search results found.</p>
            )}
          </div>
        </section>

        <section className="document-list-section">
          <h2>All Documents</h2>
          {loading ? (
            <p>Loading documents...</p>
          ) : error ? (
            <p style={{color: 'red'}}>{error}</p>
          ) : (
            <ul>
              {Array.isArray(documents) && documents.length > 0 ? (
                documents.map((doc) => (
                  <li key={doc.id} onClick={() => handleDocumentClick(doc.id)}>
                    {doc.filename} (ID: {doc.id}) - {doc.classification_category}
                  </li>
                ))
              ) : (
                <p>No documents found.</p>
              )}
            </ul>
          )}
        </section>

        {selectedDocumentId && documentDetails && (
          <section className="document-details-section">
            <h2>Details for Document ID: {selectedDocumentId}</h2>
            <h3>Filename: {documentDetails.filename}</h3>
            <p><strong>File Type:</strong> {documentDetails.file_type}</p>
            <p><strong>Classification:</strong> {documentDetails.classification_category} (Score: {documentDetails.classification_score.toFixed(2)})</p>
            <p><strong>Status:</strong> {documentDetails.status}</p>
            <p><strong>Processing Timestamp:</strong> {new Date(documentDetails.processing_timestamp).toLocaleString()}</p>

            <h3>Extracted Text:</h3>
            <textarea value={documentDetails.extracted_text} readOnly rows="10" cols="80"></textarea>

            <h3>Extracted Entities:</h3>
            {documentDetails.entities && documentDetails.entities.length > 0 ? (
              <ul>
                {documentDetails.entities.map((entity, index) => (
                  <li key={index}>
                    <strong>{entity.entity_type}:</strong> {entity.text} (Score: {entity.score.toFixed(2)})
                  </li>
                ))}
              </ul>
            ) : <p>No entities extracted.</p>}

            <h3>Key Phrases:</h3>
            {documentDetails.key_phrases && documentDetails.key_phrases.length > 0 ? (
              <ul>
                {documentDetails.key_phrases.map((phrase, index) => (
                  <li key={index}>{phrase.phrase}</li>
                ))}
              </ul>
            ) : <p>No key phrases extracted.</p>}

            <h3>Document Sections:</h3>
            {documentDetails.document_sections && Object.keys(documentDetails.document_sections).length > 0 ? (
              <ul>
                {Object.entries(documentDetails.document_sections).map(([title, content]) => (
                  <li key={title}>
                    <strong>{title}:</strong> {content.substring(0, 150)}...
                  </li>
                ))}
              </ul>
            ) : <p>No sections extracted.</p>}

            <h3>Structured Data:</h3>
            <h4>Equipment:</h4>
            {documentDetails.equipment && documentDetails.equipment.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Specifications</th>
                    <th>Location</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {documentDetails.equipment.map((item, index) => (
                    <tr key={index}>
                      <td>{item.name}</td>
                      <td>{item.type}</td>
                      <td>{JSON.stringify(item.specifications)}</td>
                      <td>{item.location || 'N/A'}</td>
                      <td>{item.confidence ? item.confidence.toFixed(2) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p>No equipment data extracted.</p>}

            <h4>Procedures:</h4>
            {documentDetails.procedures && documentDetails.procedures.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Title</th>
                    <th>Category</th>
                    <th>Steps Count</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {documentDetails.procedures.map((proc, index) => (
                    <tr key={index}>
                      <td>{proc.title}</td>
                      <td>{proc.category || 'N/A'}</td>
                      <td>{proc.steps.length}</td>
                      <td>{proc.confidence ? proc.confidence.toFixed(2) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p>No procedure data extracted.</p>}

            <h4>Safety Information:</h4>
            {documentDetails.safety_info && documentDetails.safety_info.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Hazard</th>
                    <th>Precaution</th>
                    <th>PPE Required</th>
                    <th>Severity</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {documentDetails.safety_info.map((safety, index) => (
                    <tr key={index}>
                      <td>{safety.hazard}</td>
                      <td>{safety.precaution}</td>
                      <td>{safety.ppe_required}</td>
                      <td>{safety.severity || 'N/A'}</td>
                      <td>{safety.confidence ? safety.confidence.toFixed(2) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p>No safety information extracted.</p>}

            <h4>Technical Specifications:</h4>
            {documentDetails.technical_specs && documentDetails.technical_specs.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Unit</th>
                    <th>Tolerance</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {documentDetails.technical_specs.map((spec, index) => (
                    <tr key={index}>
                      <td>{spec.parameter}</td>
                      <td>{spec.value}</td>
                      <td>{spec.unit || 'N/A'}</td>
                      <td>{spec.tolerance || 'N/A'}</td>
                      <td>{spec.confidence ? spec.confidence.toFixed(2) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p>No technical specifications extracted.</p>}

            <h4>Personnel:</h4>
            {documentDetails.personnel && documentDetails.personnel.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Role</th>
                    <th>Responsibilities</th>
                    <th>Certifications</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {documentDetails.personnel.map((person, index) => (
                    <tr key={index}>
                      <td>{person.name}</td>
                      <td>{person.role}</td>
                      <td>{person.responsibilities || 'N/A'}</td>
                      <td>{person.certifications ? person.certifications.join(', ') : 'N/A'}</td>
                      <td>{person.confidence ? person.confidence.toFixed(2) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <p>No personnel data extracted.</p>}

          </section>
        )}
      </main>
    </div>
  );
}

export default App;
