import { Github, BookOpen, Menu, X } from 'lucide-react';
import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface NavbarProps {
  activeTab: 'hide' | 'extract';
  onTabChange: (tab: 'hide' | 'extract') => void;
}

export const Navbar = ({ activeTab, onTabChange }: NavbarProps) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="fixed top-0 left-0 right-0 z-50 bg-navy-900/95 backdrop-blur-sm border-b border-white/5"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-teal to-teal-dark flex items-center justify-center">
              <span className="text-navy-900 font-bold text-sm">S</span>
            </div>
            <span className="font-heading font-semibold text-lg hidden sm:block">
              Steganography
            </span>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            <button
              onClick={() => onTabChange('hide')}
              className={`px-4 py-2 rounded-lg transition-all duration-150 ${
                activeTab === 'hide'
                  ? 'bg-teal/10 text-teal'
                  : 'text-white/70 hover:text-white hover:bg-white/5'
              }`}
            >
              Hide Secret
            </button>
            <button
              onClick={() => onTabChange('extract')}
              className={`px-4 py-2 rounded-lg transition-all duration-150 ${
                activeTab === 'extract'
                  ? 'bg-teal/10 text-teal'
                  : 'text-white/70 hover:text-white hover:bg-white/5'
              }`}
            >
              Extract Secret
            </button>
          </div>

          {/* External Links */}
          <div className="hidden md:flex items-center gap-2">
            <a
              href="https://github.com/Tejasvi-hegde/Steganography"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-white/70 hover:text-white hover:bg-white/5 rounded-lg transition-all"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="#about"
              className="p-2 text-white/70 hover:text-white hover:bg-white/5 rounded-lg transition-all"
            >
              <BookOpen className="w-5 h-5" />
            </a>
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 text-white/70 hover:text-white"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-navy-800 border-b border-white/5"
          >
            <div className="px-4 py-4 space-y-2">
              <button
                onClick={() => {
                  onTabChange('hide');
                  setMobileMenuOpen(false);
                }}
                className={`w-full text-left px-4 py-2 rounded-lg ${
                  activeTab === 'hide' ? 'bg-teal/10 text-teal' : 'text-white/70'
                }`}
              >
                Hide Secret
              </button>
              <button
                onClick={() => {
                  onTabChange('extract');
                  setMobileMenuOpen(false);
                }}
                className={`w-full text-left px-4 py-2 rounded-lg ${
                  activeTab === 'extract' ? 'bg-teal/10 text-teal' : 'text-white/70'
                }`}
              >
                Extract Secret
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
};
