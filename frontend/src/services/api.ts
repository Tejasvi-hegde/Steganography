import axios from 'axios';
import { ProcessingResult, ExtractResult, HealthCheckResponse, RobustnessTestResponse } from '../types';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 120 seconds for processing (robustness test takes longer)
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

  /**
   * Test robustness of stego image against JPEG compression
   */
  async robustnessTest(stegoImage: string): Promise<RobustnessTestResponse> {
    // Convert base64 to blob
    const base64Data = stegoImage.split(',')[1];
    const byteCharacters = atob(base64Data);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/png' });
    
    const formData = new FormData();
    formData.append('stego', blob, 'stego.png');

    const response = await api.post('/robustness-test', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to run robustness test');
    }

    return response.data.data;
  },

  /**
   * Generate a PDF report of the analysis
   */
  async generateReport(metrics: Record<string, number>): Promise<Blob> {
    const response = await api.post('/generate-report', { metrics }, {
      responseType: 'blob',
    });
    return response.data;
  },
};
