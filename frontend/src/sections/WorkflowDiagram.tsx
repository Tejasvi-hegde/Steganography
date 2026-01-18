import { motion } from 'framer-motion';
import { Image, Cpu, ArrowRight, Eye, EyeOff } from 'lucide-react';

export const WorkflowDiagram = () => {
  const steps = [
    {
      icon: Image,
      label: 'Cover Image',
      description: 'The visible carrier',
      color: 'from-blue-500 to-blue-600',
    },
    {
      icon: EyeOff,
      label: 'Secret Image',
      description: 'Hidden within',
      color: 'from-amber to-amber-light',
    },
    {
      icon: Cpu,
      label: 'Encoder',
      description: 'Neural network',
      color: 'from-teal to-teal-light',
    },
    {
      icon: Image,
      label: 'Stego Image',
      description: 'Contains secret',
      color: 'from-purple-500 to-purple-600',
    },
    {
      icon: Cpu,
      label: 'Decoder',
      description: 'Extracts secret',
      color: 'from-teal to-teal-light',
    },
    {
      icon: Eye,
      label: 'Recovered',
      description: 'Original secret',
      color: 'from-green-500 to-green-600',
    },
  ];

  return (
    <section id="workflow" className="py-16 px-4 bg-navy-800/30">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-2xl sm:text-3xl font-heading font-bold mb-4">
            How It Works
          </h2>
          <p className="text-slate-blue max-w-xl mx-auto">
            Our deep learning model encodes a secret image into a cover image,
            creating a stego image that looks identical to the original.
          </p>
        </motion.div>

        {/* Desktop Workflow */}
        <div className="hidden lg:block">
          <div className="flex items-center justify-between gap-2">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center"
              >
                <div className="flex flex-col items-center">
                  <div
                    className={`w-16 h-16 rounded-xl bg-gradient-to-br ${step.color} 
                                flex items-center justify-center mb-3 shadow-lg`}
                  >
                    <step.icon className="w-8 h-8 text-white" />
                  </div>
                  <span className="text-sm font-medium text-white">{step.label}</span>
                  <span className="text-xs text-slate-blue">{step.description}</span>
                </div>

                {index < steps.length - 1 && (
                  <ArrowRight className="w-6 h-6 text-slate-blue mx-4 arrow-animate" />
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Mobile Workflow */}
        <div className="lg:hidden">
          <div className="grid grid-cols-2 gap-4">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.95 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.05 }}
                className="flex flex-col items-center p-4 bg-navy-800 rounded-xl"
              >
                <div
                  className={`w-12 h-12 rounded-lg bg-gradient-to-br ${step.color} 
                              flex items-center justify-center mb-2`}
                >
                  <step.icon className="w-6 h-6 text-white" />
                </div>
                <span className="text-sm font-medium text-white text-center">
                  {step.label}
                </span>
                <span className="text-xs text-slate-blue text-center">
                  {step.description}
                </span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Info Box */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
          className="mt-12 p-6 bg-navy-800 rounded-xl border border-white/5 text-center"
        >
          <p className="text-slate-blue">
            <span className="text-teal font-semibold">Model Architecture:</span>{' '}
            Encoder (1.09M params) + Decoder (783K params) = 1.88M total parameters
          </p>
        </motion.div>
      </div>
    </section>
  );
};
