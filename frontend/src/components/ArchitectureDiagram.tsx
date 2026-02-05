import { motion } from 'framer-motion';
import { Layers, ArrowRight, Box, Cpu } from 'lucide-react';

interface LayerInfo {
  name: string;
  type: string;
  inputShape: string;
  outputShape: string;
  params: string;
}

const encoderLayers: LayerInfo[] = [
  { name: 'Input (Cover)', type: 'Input', inputShape: '3×256×256', outputShape: '3×256×256', params: '0' },
  { name: 'Input (Secret)', type: 'Input', inputShape: '3×256×256', outputShape: '3×256×256', params: '0' },
  { name: 'Concat', type: 'Concatenation', inputShape: '6×256×256', outputShape: '6×256×256', params: '0' },
  { name: 'Conv Block 1', type: 'Conv2d + ReLU', inputShape: '6×256×256', outputShape: '64×256×256', params: '3,520' },
  { name: 'Conv Block 2', type: 'Conv2d + ReLU', inputShape: '64×256×256', outputShape: '128×128×128', params: '73,856' },
  { name: 'Conv Block 3', type: 'Conv2d + ReLU', inputShape: '128×128×128', outputShape: '256×64×64', params: '295,168' },
  { name: 'Conv Block 4', type: 'Conv2d + ReLU', inputShape: '256×64×64', outputShape: '512×32×32', params: '1,180,160' },
  { name: 'Deconv Block 1', type: 'ConvTranspose2d', inputShape: '512×32×32', outputShape: '256×64×64', params: '1,179,904' },
  { name: 'Deconv Block 2', type: 'ConvTranspose2d', inputShape: '256×64×64', outputShape: '128×128×128', params: '295,040' },
  { name: 'Deconv Block 3', type: 'ConvTranspose2d', inputShape: '128×128×128', outputShape: '64×256×256', params: '73,792' },
  { name: 'Output Layer', type: 'Conv2d + Tanh', inputShape: '64×256×256', outputShape: '3×256×256', params: '1,731' },
];

const decoderLayers: LayerInfo[] = [
  { name: 'Input (Stego)', type: 'Input', inputShape: '3×256×256', outputShape: '3×256×256', params: '0' },
  { name: 'Conv Block 1', type: 'Conv2d + ReLU', inputShape: '3×256×256', outputShape: '64×256×256', params: '1,792' },
  { name: 'Conv Block 2', type: 'Conv2d + ReLU', inputShape: '64×256×256', outputShape: '128×128×128', params: '73,856' },
  { name: 'Conv Block 3', type: 'Conv2d + ReLU', inputShape: '128×128×128', outputShape: '256×64×64', params: '295,168' },
  { name: 'Conv Block 4', type: 'Conv2d + ReLU', inputShape: '256×64×64', outputShape: '512×32×32', params: '1,180,160' },
  { name: 'Deconv Block 1', type: 'ConvTranspose2d', inputShape: '512×32×32', outputShape: '256×64×64', params: '1,179,904' },
  { name: 'Deconv Block 2', type: 'ConvTranspose2d', inputShape: '256×64×64', outputShape: '128×128×128', params: '295,040' },
  { name: 'Deconv Block 3', type: 'ConvTranspose2d', inputShape: '128×128×128', outputShape: '64×256×256', params: '73,792' },
  { name: 'Output Layer', type: 'Conv2d + Tanh', inputShape: '64×256×256', outputShape: '3×256×256', params: '1,731' },
];

const LayerCard = ({ layer, index, color }: { layer: LayerInfo; index: number; color: string }) => (
  <motion.div
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    transition={{ delay: index * 0.05 }}
    className={`p-3 rounded-lg border ${color} bg-navy-800/50`}
  >
    <div className="flex items-center justify-between mb-1">
      <span className="font-medium text-sm">{layer.name}</span>
      <span className="text-xs px-2 py-0.5 rounded bg-navy-700/50 text-slate-blue">{layer.type}</span>
    </div>
    <div className="grid grid-cols-3 gap-2 text-xs text-slate-blue">
      <div>
        <span className="block text-[10px] uppercase tracking-wider opacity-70">Input</span>
        <span className="font-mono">{layer.inputShape}</span>
      </div>
      <div>
        <span className="block text-[10px] uppercase tracking-wider opacity-70">Output</span>
        <span className="font-mono">{layer.outputShape}</span>
      </div>
      <div>
        <span className="block text-[10px] uppercase tracking-wider opacity-70">Params</span>
        <span className="font-mono">{layer.params}</span>
      </div>
    </div>
  </motion.div>
);

export const ArchitectureDiagram = () => {
  const totalEncoderParams = 3103171;
  const totalDecoderParams = 3101443;

  return (
    <section className="py-16 px-4 bg-navy-800/30">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          {/* Header */}
          <div className="flex items-center justify-center gap-3 mb-8">
            <Layers className="w-8 h-8 text-teal" />
            <h2 className="text-2xl font-heading font-bold">Neural Network Architecture</h2>
          </div>

          {/* Overview */}
          <div className="card mb-8">
            <h3 className="text-lg font-heading font-semibold mb-4">System Overview</h3>
            <p className="text-slate-blue mb-6">
              The steganography system uses an Encoder-Decoder GAN architecture. The Encoder hides a secret image 
              inside a cover image to produce a stego image, while the Decoder extracts the hidden secret from the stego image.
            </p>
            
            {/* Visual Flow */}
            <div className="flex items-center justify-center gap-4 flex-wrap mb-6">
              <div className="flex flex-col items-center p-4 rounded-lg bg-navy-700/50">
                <Box className="w-8 h-8 text-blue-400 mb-2" />
                <span className="text-sm font-medium">Cover Image</span>
                <span className="text-xs text-slate-blue">3×256×256</span>
              </div>
              <ArrowRight className="w-6 h-6 text-slate-blue hidden sm:block" />
              <div className="flex flex-col items-center p-4 rounded-lg bg-teal/20 border border-teal/30">
                <Cpu className="w-8 h-8 text-teal mb-2" />
                <span className="text-sm font-medium">Encoder</span>
                <span className="text-xs text-slate-blue">{totalEncoderParams.toLocaleString()} params</span>
              </div>
              <ArrowRight className="w-6 h-6 text-slate-blue hidden sm:block" />
              <div className="flex flex-col items-center p-4 rounded-lg bg-purple-500/20 border border-purple-500/30">
                <Box className="w-8 h-8 text-purple-400 mb-2" />
                <span className="text-sm font-medium">Stego Image</span>
                <span className="text-xs text-slate-blue">3×256×256</span>
              </div>
              <ArrowRight className="w-6 h-6 text-slate-blue hidden sm:block" />
              <div className="flex flex-col items-center p-4 rounded-lg bg-amber/20 border border-amber/30">
                <Cpu className="w-8 h-8 text-amber mb-2" />
                <span className="text-sm font-medium">Decoder</span>
                <span className="text-xs text-slate-blue">{totalDecoderParams.toLocaleString()} params</span>
              </div>
              <ArrowRight className="w-6 h-6 text-slate-blue hidden sm:block" />
              <div className="flex flex-col items-center p-4 rounded-lg bg-navy-700/50">
                <Box className="w-8 h-8 text-green-400 mb-2" />
                <span className="text-sm font-medium">Recovered Secret</span>
                <span className="text-xs text-slate-blue">3×256×256</span>
              </div>
            </div>

            {/* Also takes Secret */}
            <div className="flex justify-center mb-4">
              <div className="flex items-center gap-2 p-3 rounded-lg bg-navy-700/30">
                <Box className="w-6 h-6 text-rose-400" />
                <span className="text-sm">Secret Image (3×256×256) → Encoder</span>
              </div>
            </div>
          </div>

          {/* Architecture Details */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Encoder */}
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 rounded-lg bg-teal/20">
                  <Cpu className="w-5 h-5 text-teal" />
                </div>
                <div>
                  <h3 className="font-heading font-semibold">Encoder Network</h3>
                  <p className="text-xs text-slate-blue">{totalEncoderParams.toLocaleString()} trainable parameters</p>
                </div>
              </div>
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                {encoderLayers.map((layer, idx) => (
                  <LayerCard key={idx} layer={layer} index={idx} color="border-teal/20" />
                ))}
              </div>
            </div>

            {/* Decoder */}
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 rounded-lg bg-amber/20">
                  <Cpu className="w-5 h-5 text-amber" />
                </div>
                <div>
                  <h3 className="font-heading font-semibold">Decoder Network</h3>
                  <p className="text-xs text-slate-blue">{totalDecoderParams.toLocaleString()} trainable parameters</p>
                </div>
              </div>
              <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                {decoderLayers.map((layer, idx) => (
                  <LayerCard key={idx} layer={layer} index={idx} color="border-amber/20" />
                ))}
              </div>
            </div>
          </div>

          {/* Technical Details */}
          <div className="card mt-6">
            <h3 className="text-lg font-heading font-semibold mb-4">Training Details</h3>
            <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-3 rounded-lg bg-navy-700/30">
                <span className="block text-xs uppercase tracking-wider text-slate-blue mb-1">Loss Function</span>
                <span className="font-medium">MSE + Perceptual</span>
              </div>
              <div className="p-3 rounded-lg bg-navy-700/30">
                <span className="block text-xs uppercase tracking-wider text-slate-blue mb-1">Optimizer</span>
                <span className="font-medium">Adam (lr=0.0002)</span>
              </div>
              <div className="p-3 rounded-lg bg-navy-700/30">
                <span className="block text-xs uppercase tracking-wider text-slate-blue mb-1">Batch Size</span>
                <span className="font-medium">16</span>
              </div>
              <div className="p-3 rounded-lg bg-navy-700/30">
                <span className="block text-xs uppercase tracking-wider text-slate-blue mb-1">Image Size</span>
                <span className="font-medium">256×256 RGB</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};
