import React, { useState, useMemo } from 'react';

// Utility function to format specifications as bullet points
const formatSpecifications = (specs) => {
  if (!specs) return 'N/A';
  
  if (typeof specs === 'string') {
    try {
      specs = JSON.parse(specs);
    } catch {
      return specs; // Return as-is if not JSON
    }
  }
  
  if (typeof specs === 'object' && specs !== null) {
    const entries = Object.entries(specs);
    if (entries.length === 0) return 'N/A';
    
    // Format keys to be more readable
    const formatKey = (key) => {
      return key
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/\b\w/g, l => l.toUpperCase())
        .trim();
    };
    
  return entries.map(([key, value]) => `- ${formatKey(key)}: ${value}`).join('\n');
  }
  
  return specs;
};

const KnowledgeTable = ({ data, onRefresh, isProcessing }) => {
  const [filter, setFilter] = useState('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('timestamp');
  const [sortOrder, setSortOrder] = useState('desc');

  // Flatten and structure the knowledge data
  const knowledgeItems = useMemo(() => {
    const items = [];
    
    data.forEach(doc => {
      // Equipment
      doc.equipment?.forEach(item => {
        items.push({
          id: `${doc.id}-eq-${item.name}`,
          document: doc.filename,
          type: 'Equipment',
          name: item.name,
          details: item.type,
          specifications: item.specifications,
          location: item.location,
          confidence: item.confidence,
          timestamp: doc.processing_timestamp
        });
      });

      // Procedures
      doc.procedures?.forEach(proc => {
        items.push({
          id: `${doc.id}-proc-${proc.title}`,
          document: doc.filename,
          type: 'Procedure',
          name: proc.title,
          details: proc.category,
          specifications: `${proc.steps?.length || 0} steps`,
          location: null,
          confidence: proc.confidence,
          timestamp: doc.processing_timestamp
        });
      });

      // Safety Information
      doc.safety_info?.forEach(safety => {
        items.push({
          id: `${doc.id}-safety-${safety.hazard}`,
          document: doc.filename,
          type: 'Safety',
          name: safety.hazard,
          details: safety.precaution,
          specifications: safety.ppe_required,
          location: safety.severity,
          confidence: safety.confidence,
          timestamp: doc.processing_timestamp
        });
      });

      // Technical Specifications
      doc.technical_specs?.forEach(spec => {
        items.push({
          id: `${doc.id}-spec-${spec.parameter}`,
          document: doc.filename,
          type: 'Technical Spec',
          name: spec.parameter,
          details: `${spec.value} ${spec.unit || ''}`,
          specifications: spec.tolerance,
          location: null,
          confidence: spec.confidence,
          timestamp: doc.processing_timestamp
        });
      });

      // Personnel
      doc.personnel?.forEach(person => {
        items.push({
          id: `${doc.id}-person-${person.name}`,
          document: doc.filename,
          type: 'Personnel',
          name: person.name,
          details: person.role,
          specifications: person.responsibilities,
          location: person.certifications?.join(', '),
          confidence: person.confidence,
          timestamp: doc.processing_timestamp
        });
      });
    });

    return items;
  }, [data]);

  // Filter and search
  const filteredItems = useMemo(() => {
    let filtered = knowledgeItems;

    if (filter !== 'all') {
      filtered = filtered.filter(item => item.type === filter);
    }

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter(item => 
        item.name.toLowerCase().includes(term) ||
        item.details?.toLowerCase().includes(term) ||
        item.document.toLowerCase().includes(term)
      );
    }

    // Sort
    filtered.sort((a, b) => {
      let aVal = a[sortBy];
      let bVal = b[sortBy];
      
      if (sortBy === 'timestamp') {
        aVal = new Date(aVal);
        bVal = new Date(bVal);
      }
      
      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return filtered;
  }, [knowledgeItems, filter, searchTerm, sortBy, sortOrder]);

  const handleSort = (column) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('asc');
    }
  };

  const getTypeColor = (type) => {
    const colors = {
      'Equipment': '#4CAF50',
      'Procedure': '#2196F3',
      'Safety': '#FF9800',
      'Technical Spec': '#9C27B0',
      'Personnel': '#607D8B'
    };
    return colors[type] || '#666';
  };

  return (
    <div className="knowledge-table-container">
      <div className="table-controls">
        <div className="filters">
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All Types</option>
            <option value="Equipment">Equipment</option>
            <option value="Procedure">Procedures</option>
            <option value="Safety">Safety</option>
            <option value="Technical Spec">Technical Specs</option>
            <option value="Personnel">Personnel</option>
          </select>
          
          <input
            type="text"
            placeholder="Search knowledge..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          
          <button onClick={onRefresh} className="refresh-btn">
            Refresh
          </button>
        </div>
        
        <div className="stats">
          <span>Total: {filteredItems.length} items</span>
        </div>
      </div>

      <div className="table-wrapper">
        <table className="knowledge-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('type')} className="sortable">
                Type {sortBy === 'type' && (sortOrder === 'asc' ? '(asc)' : '(desc)')}
              </th>
              <th onClick={() => handleSort('name')} className="sortable">
                Name {sortBy === 'name' && (sortOrder === 'asc' ? '(asc)' : '(desc)')}
              </th>
              <th>Details</th>
              <th>Specifications</th>
              <th>Location/Severity</th>
              <th onClick={() => handleSort('confidence')} className="sortable">
                Confidence {sortBy === 'confidence' && (sortOrder === 'asc' ? '(asc)' : '(desc)')}
              </th>
              <th onClick={() => handleSort('document')} className="sortable">
                Source Document {sortBy === 'document' && (sortOrder === 'asc' ? '(asc)' : '(desc)')}
              </th>
              <th onClick={() => handleSort('timestamp')} className="sortable">
                Extracted {sortBy === 'timestamp' && (sortOrder === 'asc' ? '(asc)' : '(desc)')}
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredItems.map((item) => (
              <tr key={item.id}>
                <td>
                  <span 
                    className="type-badge" 
                    style={{ backgroundColor: getTypeColor(item.type) }}
                  >
                    {item.type}
                  </span>
                </td>
                <td className="name-cell">{item.name}</td>
                <td className="details-cell">{item.details || 'N/A'}</td>
                <td className="specs-cell">
                  <div className="specs-content">
                    {formatSpecifications(item.specifications)}
                  </div>
                </td>
                <td>{item.location || 'N/A'}</td>
                <td className="confidence-cell">
                  {item.confidence ? 
                    <span className={`confidence ${item.confidence > 0.8 ? 'high' : item.confidence > 0.6 ? 'medium' : 'low'}`}>
                      {(item.confidence * 100).toFixed(0)}%
                    </span> 
                    : 'N/A'
                  }
                </td>
                <td className="document-cell">{item.document}</td>
                <td className="timestamp-cell">
                  {new Date(item.timestamp).toLocaleDateString()}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        
        {filteredItems.length === 0 && (
          <div className="no-data">
            {isProcessing ? (
              <div>
                <h3>Processing documents...</h3>
                <p>Knowledge extraction is in progress. This may take a few minutes.</p>
                <p>The table will automatically update when processing is complete.</p>
              </div>
            ) : data.length === 0 ? (
              <div>
                <h3>No documents uploaded yet</h3>
                <p>Upload documents above to start extracting knowledge from your enterprise documents.</p>
              </div>
            ) : (
              <div>
                <h3>No knowledge extracted</h3>
                <p>Documents have been processed but no structured knowledge was found.</p>
                <p>Try uploading documents with more structured content.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default KnowledgeTable;