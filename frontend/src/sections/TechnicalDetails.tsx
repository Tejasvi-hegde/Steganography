import { motion } from 'framer-motion';
import { ChevronDown, ChevronUp, Cpu, Monitor } from 'lucide-react';
import { useState } from 'react';

interface TechnicalDetailsProps {
  processingTime?: number;
}

export const TechnicalDetails = ({ processingTime }: TechnicalDetailsProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <section className="py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="card"
        >
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="w-full flex items-center justify-between text-left"
          >
            <div className="flex items-center gap-3">
              <Cpu className="w-5 h-5 text-teal" />
              <span className="font-heading font-semibold">Technical Details</span>
            </div>
            {isExpanded ? (
              <ChevronUp className="w-5 h-5 text-slate-blue" />
            ) : (
              <ChevronDown className="w-5 h-5 text-slate-blue" />
            )}
          </button>

          {isExpanded && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6 grid md:grid-cols-2 gap-6"
            >
              {/* Model Architecture */}
              <div className="bg-navy-900/50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-teal" />
                  Model Architecture
                </h4>
                <div className="space-y-3 font-mono text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-blue">Encoder</span>
                    <span className="text-white">1,097,795 params</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-blue">Decoder</span>
                    <span className="text-white">783,619 params</span>
                  </div>
                  <div className="border-t border-white/10 pt-3 flex justify-between">
                    <span className="text-slate-blue">Total</span>
                    <span className="text-teal font-semibold">1,881,414 params</span>
                  </div>
                </div>
              </div>

              {/* Processing Info */}
              <div className="bg-navy-900/50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                  <Monitor className="w-4 h-4 text-teal" />
                  Processing Info
                </h4>
                <div className="space-y-3 font-mono text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-blue">Input Size</span>
                    <span className="text-white">128 × 128 px</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-blue">Output Format</span>
                    <span className="text-white">PNG (lossless)</span>
                  </div>
                  {processingTime && (
                    <div className="flex justify-between">
                      <span className="text-slate-blue">Last Processing</span>
                      <span className="text-white">{processingTime.toFixed(2)}s</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-slate-blue">Framework</span>
                    <span className="text-white">PyTorch + DirectML</span>
                  </div>
                </div>
              </div>

              {/* Architecture Description */}
              <div className="md:col-span-2 bg-navy-900/50 rounded-lg p-4">
                <h4 className="text-sm font-semibold text-white mb-3">Network Details</h4>
                <p className="text-sm text-slate-blue leading-relaxed">
                  <strong className="text-white">Encoder:</strong> Takes 6-channel input (cover + secret), 
                  uses residual blocks with skip connections for feature learning, outputs 3-channel stego image.
                  <br /><br />
                  <strong className="text-white">Decoder:</strong> Takes 3-channel stego image, 
                  extracts hidden information through convolutional layers, recovers the 3-channel secret image.
                </p>
              </div>
            </motion.div>
          )}
        </motion.div>
      </div>
    </section>
  );
};
