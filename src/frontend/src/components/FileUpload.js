import React, { useState } from 'react';
import { apiService } from '../services/apiService';

const FileUpload = ({ onUploadSuccess }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (event) => {
    const files = Array.from(event.target.files);
    setSelectedFiles(files);
    setMessage('');
  };

  const handleUpload = async () => {
    if (selectedFiles.length === 0) {
      setMessage('Please select files to upload');
      return;
    }

    setUploading(true);
    setMessage('Uploading and processing files...');

    try {
      const results = await Promise.all(
        selectedFiles.map(file => apiService.uploadFile(file))
      );
      
      const successCount = results.filter(r => r.success).length;
      setMessage(`Successfully uploaded ${successCount}/${selectedFiles.length} files. Processing in background.`);
      setSelectedFiles([]);
      
      // Clear file input
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) fileInput.value = '';
      
      onUploadSuccess();
    } catch (error) {
      setMessage(`Upload failed: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    setSelectedFiles(files);
    setMessage('');
  };

  const handleDragOver = (event) => {
    event.preventDefault();
  };

  return (
    <div className="upload-section">
      <h2>Upload Documents & Media</h2>
      <div 
        className="upload-area"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input 
          type="file" 
          multiple 
          onChange={handleFileChange}
          disabled={uploading}
          accept=".pdf,.doc,.docx,.txt,.md,.mp4,.avi,.mov,.mkv,.jpg,.jpeg,.png"
        />
        <div className="upload-info">
          <p>Drop files here or click to select</p>
          <p className="file-types">Supports: PDF, DOC, DOCX, TXT, MD, MP4, AVI, MOV, MKV, JPG, PNG</p>
        </div>
      </div>
      
      {selectedFiles.length > 0 && (
        <div className="selected-files">
          <h3>Selected Files ({selectedFiles.length}):</h3>
          <ul>
            {selectedFiles.map((file, index) => (
              <li key={index}>{file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)</li>
            ))}
          </ul>
        </div>
      )}
      
      <button 
        onClick={handleUpload} 
        disabled={uploading || selectedFiles.length === 0}
        className="upload-btn"
      >
        {uploading ? 'Processing...' : `Upload ${selectedFiles.length} File${selectedFiles.length !== 1 ? 's' : ''}`}
      </button>
      
      {message && <div className={`message ${message.includes('failed') ? 'error' : 'success'}`}>{message}</div>}
    </div>
  );
};

export default FileUpload;