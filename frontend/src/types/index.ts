// Application State Types
export interface ImageData {
  file: File | null;
  preview: string | null;
  name: string | null;
}

export interface Metrics {
  psnrStego: number;
  ssimStego: number;
  psnrRecovery: number;
  mse: number;
  processingTime: number;
}

export interface ProcessingResult {
  stegoImage: string;
  recoveredSecret: string;
  metrics: Metrics;
}

export interface ExtractResult {
  recoveredSecret: string;
  processingTime: number;
}

export interface AppState {
  activeTab: 'hide' | 'extract';
  coverImage: ImageData;
  secretImage: ImageData;
  stegoImage: ImageData;
  isProcessing: boolean;
  processingProgress: number;
  results: ProcessingResult | null;
  extractResults: ExtractResult | null;
  error: string | null;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
  encoder_params: number;
  decoder_params: number;
  device: string;
}

// Component Props Types
export interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export interface ImageUploaderProps {
  label: string;
  image: ImageData;
  onImageSelect: (file: File) => void;
  onImageClear: () => void;
  accept?: string;
}

export interface MetricCardProps {
  label: string;
  value: number;
  unit?: string;
  quality?: 'excellent' | 'good' | 'fair' | 'poor';
}

export interface ResultImageProps {
  label: string;
  src: string;
  downloadName?: string;
}
