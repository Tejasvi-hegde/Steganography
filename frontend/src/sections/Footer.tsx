import { Github, Heart } from 'lucide-react';

export const Footer = () => {
  return (
    <footer id="about" className="py-12 px-4 border-t border-white/5">
      <div className="max-w-6xl mx-auto">
        <div className="grid md:grid-cols-3 gap-8 mb-8">
          {/* About */}
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal to-teal-dark flex items-center justify-center">
                <span className="text-navy-900 font-bold text-sm">S</span>
              </div>
              <span className="font-heading font-semibold">Steganography</span>
            </div>
            <p className="text-sm text-slate-blue leading-relaxed">
              Deep learning-based image steganography using PyTorch. 
              Hide secret images within cover images using neural network encoding.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-heading font-semibold mb-4">Resources</h4>
            <ul className="space-y-2">
              <li>
                <a
                  href="https://github.com/Tejasvi-hegde/Steganography"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-blue hover:text-teal transition-colors"
                >
                  GitHub Repository
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/Tejasvi-hegde/Steganography#readme"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-blue hover:text-teal transition-colors"
                >
                  Documentation
                </a>
              </li>
              <li>
                <a
                  href="https://en.wikipedia.org/wiki/Steganography"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm text-slate-blue hover:text-teal transition-colors"
                >
                  Learn About Steganography
                </a>
              </li>
            </ul>
          </div>

          {/* Tech Stack */}
          <div>
            <h4 className="font-heading font-semibold mb-4">Built With</h4>
            <div className="flex flex-wrap gap-2">
              {['PyTorch', 'React', 'TypeScript', 'Tailwind', 'DirectML'].map(
                (tech) => (
                  <span
                    key={tech}
                    className="px-3 py-1 bg-navy-800 rounded-full text-xs text-slate-blue border border-white/5"
                  >
                    {tech}
                  </span>
                )
              )}
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="pt-8 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-sm text-slate-blue flex items-center gap-2">
            Made with <Heart className="w-4 h-4 text-coral" /> for academic research
          </p>
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/Tejasvi-hegde/Steganography"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-slate-blue hover:text-white transition-colors"
            >
              <Github className="w-5 h-5" />
            </a>
            <span className="text-xs text-slate-blue">MIT License</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
