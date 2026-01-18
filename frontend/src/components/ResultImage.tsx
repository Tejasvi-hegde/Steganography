import { motion } from 'framer-motion';
import { Download } from 'lucide-react';
import { ResultImageProps } from '../types';

export const ResultImage = ({ label, src, downloadName }: ResultImageProps) => {
  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = src;
    link.download = downloadName || `${label.toLowerCase().replace(/\s/g, '_')}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col gap-2"
    >
      <div className="relative group">
        <img
          src={src}
          alt={label}
          className="w-full aspect-square object-cover rounded-lg border border-white/10"
        />
        {downloadName && (
          <button
            onClick={handleDownload}
            className="absolute bottom-2 right-2 p-2 bg-navy-900/80 rounded-lg
                       opacity-0 group-hover:opacity-100 transition-opacity duration-200
                       hover:bg-teal hover:text-navy-900"
          >
            <Download className="w-4 h-4" />
          </button>
        )}
      </div>
      <span className="text-xs text-center text-slate-blue">{label}</span>
    </motion.div>
  );
};
