import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Image as ImageIcon } from 'lucide-react';
import { ImageUploaderProps } from '../types';

export const ImageUploader = ({
  label,
  image,
  onImageSelect,
  onImageClear,
}: ImageUploaderProps) => {
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: any[]) => {
      setError(null);
      
      if (rejectedFiles.length > 0) {
        setError('Invalid file type. Please upload an image.');
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
          setError('File too large. Maximum size is 10MB.');
          return;
        }

        onImageSelect(file);
      }
    },
    [onImageSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] },
    maxFiles: 1,
    multiple: false,
  });

  const hasImage = image.preview !== null;

  return (
    <div className="flex flex-col gap-3">
      <label className="text-sm font-medium text-slate-blue uppercase tracking-wider">
        {label}
      </label>

      <div
        {...getRootProps()}
        className={`
          relative w-full aspect-square max-w-[280px] mx-auto
          ${hasImage ? 'cursor-default' : 'drop-zone'}
          ${isDragActive ? 'drop-zone-active' : ''}
          ${error ? 'drop-zone-error' : ''}
          overflow-hidden rounded-xl
        `}
      >
        <input {...getInputProps()} />

        <AnimatePresence mode="wait">
          {hasImage ? (
            <motion.div
              key="image"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="relative w-full h-full"
            >
              <img
                src={image.preview!}
                alt={label}
                className="w-full h-full object-cover rounded-xl"
              />
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onImageClear();
                }}
                className="absolute top-2 right-2 p-1.5 bg-navy-900/80 rounded-full
                           hover:bg-coral transition-colors duration-150"
              >
                <X className="w-4 h-4" />
              </button>
              <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-navy-900/90 to-transparent">
                <p className="text-xs text-white/80 truncate">{image.name}</p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="placeholder"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center h-full gap-3 p-4"
            >
              {isDragActive ? (
                <>
                  <motion.div
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ repeat: Infinity, duration: 1 }}
                  >
                    <Upload className="w-10 h-10 text-teal" />
                  </motion.div>
                  <p className="text-teal font-medium">Release to upload</p>
                </>
              ) : (
                <>
                  <ImageIcon className="w-10 h-10 text-slate-blue" />
                  <div className="text-center">
                    <p className="text-sm text-white/70">
                      Drop image here or{' '}
                      <span className="text-teal">browse</span>
                    </p>
                    <p className="text-xs text-slate-blue mt-1">
                      PNG, JPG up to 10MB
                    </p>
                  </div>
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="text-coral text-sm text-center"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
};
