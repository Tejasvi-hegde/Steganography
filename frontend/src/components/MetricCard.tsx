import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { MetricCardProps } from '../types';

const getQualityColor = (quality?: string) => {
  switch (quality) {
    case 'excellent':
      return 'text-teal';
    case 'good':
      return 'text-green-400';
    case 'fair':
      return 'text-amber';
    case 'poor':
      return 'text-coral';
    default:
      return 'text-white';
  }
};

const getQualityLabel = (quality?: string) => {
  switch (quality) {
    case 'excellent':
      return '● Excellent';
    case 'good':
      return '● Good';
    case 'fair':
      return '● Fair';
    case 'poor':
      return '● Needs Improvement';
    default:
      return '';
  }
};

export const MetricCard = ({ label, value, unit = '', quality }: MetricCardProps) => {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const duration = 500;
    const steps = 30;
    const stepValue = value / steps;
    const stepDuration = duration / steps;
    let current = 0;

    const interval = setInterval(() => {
      current += stepValue;
      if (current >= value) {
        setDisplayValue(value);
        clearInterval(interval);
      } else {
        setDisplayValue(current);
      }
    }, stepDuration);

    return () => clearInterval(interval);
  }, [value]);

  const formattedValue = unit === 'dB' || unit === ''
    ? displayValue.toFixed(2)
    : displayValue.toFixed(4);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="metric-card"
    >
      <span className="text-xs text-slate-blue uppercase tracking-wider mb-2">
        {label}
      </span>
      <span className="font-mono text-2xl font-semibold text-white">
        {formattedValue}
        {unit && <span className="text-sm ml-1 text-slate-blue">{unit}</span>}
      </span>
      {quality && (
        <span className={`text-xs mt-2 ${getQualityColor(quality)}`}>
          {getQualityLabel(quality)}
        </span>
      )}
    </motion.div>
  );
};
