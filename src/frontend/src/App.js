import React, { useState, useEffect } from 'react';
import './App.css';
import { FileUpload, KnowledgeTable } from './components';
import { apiService } from './services/apiService';

function App() {
  const [knowledgeData, setKnowledgeData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [hasUploadedFiles, setHasUploadedFiles] = useState(false);

  useEffect(() => {
    checkBackendAndLoadData();
  }, []);

  const checkBackendAndLoadData = async () => {
    const isBackendUp = await apiService.checkBackendStatus();
    if (!isBackendUp) {
      setError("Backend service is not available. Please ensure the API server is running on http://localhost:8000");
      setLoading(false);
      return;
    }
    loadKnowledgeData();
  };

  const loadKnowledgeData = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiService.getAllKnowledgeData();
      setKnowledgeData(data);
      
      // Check if any documents are still processing
      const processingDocs = data.filter(doc => doc.status === 'processing' || doc.status === 'pending');
      if (processingDocs.length > 0) {
        setProcessingStatus(`${processingDocs.length} document${processingDocs.length > 1 ? 's' : ''} still processing...`);
      } else {
        setProcessingStatus(null);
        // Reset upload flag when we have data or processing is complete
        if (data.length > 0) {
          setHasUploadedFiles(false);
        }
      }
    } catch (error) {
      console.error("Error loading knowledge data:", error);
      setError(`Failed to load data: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadSuccess = () => {
    setHasUploadedFiles(true);
    
    // Refresh data after successful upload and set up polling
    setTimeout(loadKnowledgeData, 2000);
    
    // Poll for updates every 10 seconds for 2 minutes after upload
    let pollCount = 0;
    const maxPolls = 12; // 2 minutes
    const pollInterval = setInterval(() => {
      pollCount++;
      loadKnowledgeData();
      
      if (pollCount >= maxPolls) {
        clearInterval(pollInterval);
      }
    }, 10000);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Enterprise Knowledge Extraction System</h1>
        <p>Upload documents • Extract knowledge • Query insights</p>
      </header>
      
      <main className="main-content">
        <FileUpload onUploadSuccess={handleUploadSuccess} />
        
        <div className="knowledge-section">
          <h2>Extracted Knowledge Database</h2>
          {processingStatus && (
            <div className="processing-status">
              <span className="processing-indicator">⏳</span>
              {processingStatus}
            </div>
          )}
          {loading ? (
            <div className="loading">Loading knowledge data...</div>
          ) : error ? (
            <div className="error">{error}</div>
          ) : (
            <KnowledgeTable 
              data={knowledgeData} 
              onRefresh={loadKnowledgeData} 
              isProcessing={!!processingStatus || hasUploadedFiles}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
