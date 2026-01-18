import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, X } from 'lucide-react';
import {
  Navbar,
  HeroSection,
  WorkflowDiagram,
  ApplicationPanel,
  ResultsSection,
  TechnicalDetails,
  Footer,
} from './sections';
import { steganographyApi } from './services';
import { ImageData, ProcessingResult, ExtractResult } from './types';

const initialImageData: ImageData = {
  file: null,
  preview: null,
  name: null,
};

function App() {
  // State
  const [activeTab, setActiveTab] = useState<'hide' | 'extract'>('hide');
  const [coverImage, setCoverImage] = useState<ImageData>(initialImageData);
  const [secretImage, setSecretImage] = useState<ImageData>(initialImageData);
  const [stegoImage, setStegoImage] = useState<ImageData>(initialImageData);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [results, setResults] = useState<ProcessingResult | null>(null);
  const [extractResults, setExtractResults] = useState<ExtractResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Image handlers
  const createImageData = useCallback((file: File): ImageData => {
    return {
      file,
      preview: URL.createObjectURL(file),
      name: file.name,
    };
  }, []);

  const clearImageData = useCallback((imageData: ImageData) => {
    if (imageData.preview) {
      URL.revokeObjectURL(imageData.preview);
    }
    return initialImageData;
  }, []);

  const handleCoverSelect = useCallback(
    (file: File) => {
      setCoverImage((prev) => {
        clearImageData(prev);
        return createImageData(file);
      });
      setResults(null);
      setError(null);
    },
    [createImageData, clearImageData]
  );

  const handleSecretSelect = useCallback(
    (file: File) => {
      setSecretImage((prev) => {
        clearImageData(prev);
        return createImageData(file);
      });
      setResults(null);
      setError(null);
    },
    [createImageData, clearImageData]
  );

  const handleStegoSelect = useCallback(
    (file: File) => {
      setStegoImage((prev) => {
        clearImageData(prev);
        return createImageData(file);
      });
      setExtractResults(null);
      setError(null);
    },
    [createImageData, clearImageData]
  );

  const handleCoverClear = useCallback(() => {
    setCoverImage((prev) => clearImageData(prev));
    setResults(null);
  }, [clearImageData]);

  const handleSecretClear = useCallback(() => {
    setSecretImage((prev) => clearImageData(prev));
    setResults(null);
  }, [clearImageData]);

  const handleStegoClear = useCallback(() => {
    setStegoImage((prev) => clearImageData(prev));
    setExtractResults(null);
  }, [clearImageData]);

  // Processing handlers
  const simulateProgress = useCallback(() => {
    setProcessingProgress(0);
    const interval = setInterval(() => {
      setProcessingProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + Math.random() * 15;
      });
    }, 200);
    return () => clearInterval(interval);
  }, []);

  const handleHideSecret = useCallback(async () => {
    if (!coverImage.file || !secretImage.file) return;

    setIsProcessing(true);
    setError(null);
    const stopProgress = simulateProgress();

    try {
      const result = await steganographyApi.hideSecret(
        coverImage.file,
        secretImage.file
      );
      setProcessingProgress(100);
      setResults(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to hide secret. Please try again.'
      );
    } finally {
      stopProgress();
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  }, [coverImage.file, secretImage.file, simulateProgress]);

  const handleExtractSecret = useCallback(async () => {
    if (!stegoImage.file) return;

    setIsProcessing(true);
    setError(null);
    const stopProgress = simulateProgress();

    try {
      const result = await steganographyApi.extractSecret(stegoImage.file);
      setProcessingProgress(100);
      setExtractResults(result);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to extract secret. Please try again.'
      );
    } finally {
      stopProgress();
      setIsProcessing(false);
      setProcessingProgress(0);
    }
  }, [stegoImage.file, simulateProgress]);

  const handleTabChange = useCallback((tab: 'hide' | 'extract') => {
    setActiveTab(tab);
    setError(null);
  }, []);

  const scrollToApp = useCallback(() => {
    document.getElementById('app')?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  return (
    <div className="min-h-screen bg-navy-900">
      <Navbar activeTab={activeTab} onTabChange={handleTabChange} />

      <main>
        <HeroSection onGetStarted={scrollToApp} />
        <WorkflowDiagram />

        <ApplicationPanel
          activeTab={activeTab}
          coverImage={coverImage}
          secretImage={secretImage}
          stegoImage={stegoImage}
          isProcessing={isProcessing}
          processingProgress={processingProgress}
          onCoverSelect={handleCoverSelect}
          onSecretSelect={handleSecretSelect}
          onStegoSelect={handleStegoSelect}
          onCoverClear={handleCoverClear}
          onSecretClear={handleSecretClear}
          onStegoClear={handleStegoClear}
          onHideSecret={handleHideSecret}
          onExtractSecret={handleExtractSecret}
        />

        {/* Error Toast */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 50 }}
              className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50"
            >
              <div className="flex items-center gap-3 px-6 py-4 bg-coral/90 rounded-lg shadow-xl">
                <AlertCircle className="w-5 h-5 text-white" />
                <span className="text-white font-medium">{error}</span>
                <button
                  onClick={() => setError(null)}
                  className="p-1 hover:bg-white/20 rounded"
                >
                  <X className="w-4 h-4 text-white" />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <ResultsSection
          activeTab={activeTab}
          results={results}
          extractResults={extractResults}
          coverImage={coverImage}
          secretImage={secretImage}
        />

        <TechnicalDetails
          processingTime={
            results?.metrics.processingTime || extractResults?.processingTime
          }
        />
      </main>

      <Footer />
    </div>
  );
}

export default App;
