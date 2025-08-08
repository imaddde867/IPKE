const API_BASE_URL = 'http://localhost:8000';

class ApiService {
  async checkBackendStatus() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return response.ok;
    } catch (error) {
      console.error("Backend health check failed:", error);
      return false;
    }
  }

  async uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/uploadfile/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      console.error(`Error uploading ${file.name}:`, error);
      return { success: false, error: error.message };
    }
  }

  async getAllKnowledgeData() {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      const documents = data.items || data || [];
      
      if (documents.length === 0) {
        console.log("No documents found in database");
        return [];
      }
      
      // Fetch detailed knowledge data for each document
      const knowledgeData = await Promise.all(
        documents.map(async (doc) => {
          try {
            const [
              entitiesRes,
              equipmentRes,
              proceduresRes,
              safetyRes,
              techSpecsRes,
              personnelRes
            ] = await Promise.all([
              fetch(`${API_BASE_URL}/documents/${doc.id}/entities/`),
              fetch(`${API_BASE_URL}/documents/${doc.id}/equipment/`),
              fetch(`${API_BASE_URL}/documents/${doc.id}/procedures/`),
              fetch(`${API_BASE_URL}/documents/${doc.id}/safety_info/`),
              fetch(`${API_BASE_URL}/documents/${doc.id}/technical_specs/`),
              fetch(`${API_BASE_URL}/documents/${doc.id}/personnel/`)
            ]);

            // Handle 404 responses gracefully (no data extracted yet)
            const safeJsonParse = async (response) => {
              if (response.status === 404) return [];
              if (!response.ok) return [];
              try {
                return await response.json();
              } catch {
                return [];
              }
            };

            return {
              ...doc,
              entities: await safeJsonParse(entitiesRes),
              equipment: await safeJsonParse(equipmentRes),
              procedures: await safeJsonParse(proceduresRes),
              safety_info: await safeJsonParse(safetyRes),
              technical_specs: await safeJsonParse(techSpecsRes),
              personnel: await safeJsonParse(personnelRes)
            };
          } catch (error) {
            console.error(`Error fetching details for document ${doc.id}:`, error);
            return {
              ...doc,
              entities: [],
              equipment: [],
              procedures: [],
              safety_info: [],
              technical_specs: [],
              personnel: []
            };
          }
        })
      );

      return knowledgeData;
    } catch (error) {
      console.error("Error fetching knowledge data:", error);
      throw error;
    }
  }

  async searchKnowledge(query) {
    try {
      const response = await fetch(`${API_BASE_URL}/search/?query=${encodeURIComponent(query)}&field=extracted_text`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error searching knowledge:", error);
      throw error;
    }
  }
}

export const apiService = new ApiService();