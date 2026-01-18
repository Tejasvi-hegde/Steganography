import axios from 'axios';
import { ProcessingResult, ExtractResult, HealthCheckResponse } from '../types';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60 seconds for processing
});

export const steganographyApi = {
  /**
   * Check if the backend is healthy and models are loaded
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Hide a secret image within a cover image
   */
  async hideSecret(coverImage: File, secretImage: File): Promise<ProcessingResult> {
    const formData = new FormData();
    formData.append('cover', coverImage);
    formData.append('secret', secretImage);

    const response = await api.post('/hide', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to hide secret');
    }

    return response.data.data;
  },

  /**
   * Extract a secret image from a stego image
   */
  async extractSecret(stegoImage: File): Promise<ExtractResult> {
    const formData = new FormData();
    formData.append('stego', stegoImage);

    const response = await api.post('/extract', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to extract secret');
    }

    return response.data.data;
  },
};
