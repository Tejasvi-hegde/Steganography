import { motion, AnimatePresence } from 'framer-motion';
import { Lock, Unlock } from 'lucide-react';
import { Button, ImageUploader, LoadingBar } from '../components';
import { ImageData } from '../types';

interface ApplicationPanelProps {
  activeTab: 'hide' | 'extract';
  coverImage: ImageData;
  secretImage: ImageData;
  stegoImage: ImageData;
  isProcessing: boolean;
  processingProgress: number;
  onCoverSelect: (file: File) => void;
  onSecretSelect: (file: File) => void;
  onStegoSelect: (file: File) => void;
  onCoverClear: () => void;
  onSecretClear: () => void;
  onStegoClear: () => void;
  onHideSecret: () => void;
  onExtractSecret: () => void;
}

export const ApplicationPanel = ({
  activeTab,
  coverImage,
  secretImage,
  stegoImage,
  isProcessing,
  processingProgress,
  onCoverSelect,
  onSecretSelect,
  onStegoSelect,
  onCoverClear,
  onSecretClear,
  onStegoClear,
  onHideSecret,
  onExtractSecret,
}: ApplicationPanelProps) => {
  const canHide = coverImage.file && secretImage.file && !isProcessing;
  const canExtract = stegoImage.file && !isProcessing;

  return (
    <section id="app" className="py-16 px-4">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="card-elevated"
        >
          {/* Tab Header */}
          <div className="flex gap-2 mb-8 p-1 bg-navy-900 rounded-lg w-fit mx-auto">
            <button
              className={`px-6 py-2 rounded-md transition-all duration-200 ${
                activeTab === 'hide'
                  ? 'bg-teal text-navy-900 font-medium'
                  : 'text-white/70 hover:text-white'
              }`}
              disabled
            >
              <Lock className="w-4 h-4 inline mr-2" />
              Hide Secret
            </button>
            <button
              className={`px-6 py-2 rounded-md transition-all duration-200 ${
                activeTab === 'extract'
                  ? 'bg-teal text-navy-900 font-medium'
                  : 'text-white/70 hover:text-white'
              }`}
              disabled
            >
              <Unlock className="w-4 h-4 inline mr-2" />
              Extract Secret
            </button>
          </div>

          <AnimatePresence mode="wait">
            {activeTab === 'hide' ? (
              <motion.div
                key="hide"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2 }}
              >
                {/* Hide Secret View */}
                <div className="grid md:grid-cols-2 gap-8 mb-8">
                  <ImageUploader
                    label="Cover Image"
                    image={coverImage}
                    onImageSelect={onCoverSelect}
                    onImageClear={onCoverClear}
                  />
                  <ImageUploader
                    label="Secret Image"
                    image={secretImage}
                    onImageSelect={onSecretSelect}
                    onImageClear={onSecretClear}
                  />
                </div>

                {isProcessing && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mb-8"
                  >
                    <LoadingBar
                      progress={processingProgress}
                      label="Encoding secret into cover image..."
                    />
                  </motion.div>
                )}

                <div className="flex justify-center">
                  <Button
                    onClick={onHideSecret}
                    disabled={!canHide}
                    loading={isProcessing}
                    className="min-w-[200px]"
                  >
                    <Lock className="w-5 h-5" />
                    Hide Secret
                  </Button>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="extract"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.2 }}
              >
                {/* Extract Secret View */}
                <div className="max-w-[300px] mx-auto mb-8">
                  <ImageUploader
                    label="Stego Image"
                    image={stegoImage}
                    onImageSelect={onStegoSelect}
                    onImageClear={onStegoClear}
                  />
                </div>

                {isProcessing && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="mb-8"
                  >
                    <LoadingBar
                      progress={processingProgress}
                      label="Extracting secret from stego image..."
                    />
                  </motion.div>
                )}

                <div className="flex justify-center">
                  <Button
                    onClick={onExtractSecret}
                    disabled={!canExtract}
                    loading={isProcessing}
                    className="min-w-[200px]"
                  >
                    <Unlock className="w-5 h-5" />
                    Extract Secret
                  </Button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </section>
  );
};
