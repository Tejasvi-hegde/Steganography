import { motion } from 'framer-motion';
import { Shield, Lock, Eye } from 'lucide-react';
import { Button } from '../components';

interface HeroSectionProps {
  onGetStarted: () => void;
}

export const HeroSection = ({ onGetStarted }: HeroSectionProps) => {
  return (
    <section className="pt-24 pb-16 px-4">
      <div className="max-w-4xl mx-auto text-center">
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="inline-flex items-center gap-2 px-4 py-2 bg-teal/10 rounded-full mb-8"
        >
          <Shield className="w-4 h-4 text-teal" />
          <span className="text-sm text-teal">Deep Learning Powered</span>
        </motion.div>

        {/* Headline */}
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-4xl sm:text-5xl lg:text-6xl font-heading font-bold mb-6"
        >
          Hide Secrets in{' '}
          <span className="gradient-text">Plain Sight</span>
        </motion.h1>

        {/* Subheadline */}
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="text-lg sm:text-xl text-slate-blue max-w-2xl mx-auto mb-10"
        >
          Advanced image steganography using neural networks. 
          Embed secret images within cover images, invisible to the human eye.
        </motion.p>

        {/* CTA Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <Button onClick={onGetStarted}>
            Get Started
          </Button>
          <Button variant="secondary" onClick={() => document.getElementById('workflow')?.scrollIntoView({ behavior: 'smooth' })}>
            How It Works
          </Button>
        </motion.div>

        {/* Feature Pills */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="flex flex-wrap justify-center gap-4 mt-12"
        >
          {[
            { icon: Lock, text: 'Secure Encoding' },
            { icon: Eye, text: 'Visually Undetectable' },
            { icon: Shield, text: 'High Quality Recovery' },
          ].map((feature, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-4 py-2 bg-navy-800 rounded-lg border border-white/5"
            >
              <feature.icon className="w-4 h-4 text-teal" />
              <span className="text-sm text-white/80">{feature.text}</span>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
};
