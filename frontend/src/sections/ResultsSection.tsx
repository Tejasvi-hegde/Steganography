import { useState } from 'react';
import { motion } from 'framer-motion';
import { Download, CheckCircle, Info, FileText, TestTube2, BarChart3 } from 'lucide-react';
import { MetricCard, ResultImage, Button } from '../components';
import { ProcessingResult, ExtractResult, ImageData, RobustnessResult } from '../types';
import { steganographyApi } from '../services';

interface ResultsSectionProps {
  activeTab: 'hide' | 'extract';
  results: ProcessingResult | null;
  extractResults: ExtractResult | null;
  coverImage: ImageData;
  secretImage: ImageData;
}

const getQuality = (psnr: number): 'excellent' | 'good' | 'fair' | 'poor' => {
  if (psnr >= 30) return 'excellent';
  if (psnr >= 25) return 'good';
  if (psnr >= 20) return 'fair';
  return 'poor';
};

const getSSIMQuality = (ssim: number): 'excellent' | 'good' | 'fair' | 'poor' => {
  if (ssim >= 0.95) return 'excellent';
  if (ssim >= 0.85) return 'good';
  if (ssim >= 0.7) return 'fair';
  return 'poor';
};

export const ResultsSection = ({
  activeTab,
  results,
  extractResults,
  coverImage,
  secretImage,
}: ResultsSectionProps) => {
  const [robustnessResults, setRobustnessResults] = useState<RobustnessResult[] | null>(null);
  const [isTestingRobustness, setIsTestingRobustness] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [showHistogram, setShowHistogram] = useState(false);

  const hasHideResults = activeTab === 'hide' && results !== null;
  const hasExtractResults = activeTab === 'extract' && extractResults !== null;

  if (!hasHideResults && !hasExtractResults) return null;

  const overallQuality = hasHideResults
    ? results!.metrics.psnrStego >= 28 && results!.metrics.ssimStego >= 0.9
      ? 'EXCELLENT'
      : results!.metrics.psnrStego >= 25
      ? 'GOOD'
      : 'FAIR'
    : null;

  const handleRobustnessTest = async () => {
    if (!results) return;
    setIsTestingRobustness(true);
    try {
      const response = await steganographyApi.robustnessTest(results.stegoImage);
      setRobustnessResults(response.results);
    } catch (error) {
      console.error('Robustness test failed:', error);
    } finally {
      setIsTestingRobustness(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!results) return;
    setIsGeneratingReport(true);
    try {
      const blob = await steganographyApi.generateReport(results.metrics);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'steganography_report.pdf';
      link.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Report generation failed:', error);
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <section className="py-16 px-4 bg-navy-800/30">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          {/* Success Header */}
          <div className="flex items-center justify-center gap-3 mb-8">
            <CheckCircle className="w-8 h-8 text-teal" />
            <h2 className="text-2xl font-heading font-bold">
              {activeTab === 'hide' ? 'Secret Hidden Successfully' : 'Secret Extracted'}
            </h2>
          </div>

          {hasHideResults && (
            <>
              {/* Image Comparison Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                {coverImage.preview && (
                  <ResultImage
                    label="Cover (Original)"
                    src={coverImage.preview}
                  />
                )}
                <ResultImage
                  label="Stego (Output)"
                  src={results!.stegoImage}
                  downloadName="stego_image.png"
                />
                {secretImage.preview && (
                  <ResultImage
                    label="Secret (Original)"
                    src={secretImage.preview}
                  />
                )}
                <ResultImage
                  label="Recovered Secret"
                  src={results!.recoveredSecret}
                  downloadName="recovered_secret.png"
                />
              </div>

              {/* Metrics Panel */}
              <div className="card mb-8">
                <h3 className="text-lg font-heading font-semibold mb-6 text-center">
                  Quality Metrics
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                  <MetricCard
                    label="PSNR (Stego)"
                    value={results!.metrics.psnrStego}
                    unit="dB"
                    quality={getQuality(results!.metrics.psnrStego)}
                  />
                  <MetricCard
                    label="SSIM (Stego)"
                    value={results!.metrics.ssimStego}
                    quality={getSSIMQuality(results!.metrics.ssimStego)}
                  />
                  <MetricCard
                    label="PSNR (Recovery)"
                    value={results!.metrics.psnrRecovery}
                    unit="dB"
                    quality={getQuality(results!.metrics.psnrRecovery)}
                  />
                  <MetricCard
                    label="MSE"
                    value={results!.metrics.mse}
                  />
                </div>

                {/* Quality Assessment */}
                <div className={`p-4 rounded-lg ${
                  overallQuality === 'EXCELLENT' 
                    ? 'bg-teal/10 border border-teal/20' 
                    : overallQuality === 'GOOD'
                    ? 'bg-green-500/10 border border-green-500/20'
                    : 'bg-amber/10 border border-amber/20'
                }`}>
                  <div className="flex items-start gap-3">
                    <Info className={`w-5 h-5 mt-0.5 ${
                      overallQuality === 'EXCELLENT' ? 'text-teal' : 
                      overallQuality === 'GOOD' ? 'text-green-400' : 'text-amber'
                    }`} />
                    <div>
                      <p className={`font-semibold ${
                        overallQuality === 'EXCELLENT' ? 'text-teal' : 
                        overallQuality === 'GOOD' ? 'text-green-400' : 'text-amber'
                      }`}>
                        Quality Assessment: {overallQuality}
                      </p>
                      <p className="text-sm text-slate-blue mt-1">
                        {overallQuality === 'EXCELLENT'
                          ? 'Stego image is visually identical to cover. Secret hidden and recoverable with high fidelity.'
                          : overallQuality === 'GOOD'
                          ? 'Good quality steganography. Minor artifacts may be present under close inspection.'
                          : 'Acceptable quality. Some visible artifacts may be present.'}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Histogram Comparison */}
              {results!.histogramComparison && (
                <div className="card mb-8">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-heading font-semibold flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-teal" />
                      Histogram Analysis
                    </h3>
                    <Button
                      variant="secondary"
                      onClick={() => setShowHistogram(!showHistogram)}
                    >
                      {showHistogram ? 'Hide' : 'Show'} Histogram
                    </Button>
                  </div>
                  {showHistogram && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                    >
                      <img 
                        src={results!.histogramComparison} 
                        alt="Histogram Comparison"
                        className="w-full rounded-lg"
                      />
                      <p className="text-sm text-slate-blue mt-2 text-center">
                        RGB channel histogram comparison between cover and stego images.
                        Similar distributions indicate good steganographic quality.
                      </p>
                    </motion.div>
                  )}
                </div>
              )}

              {/* Robustness Test Section */}
              <div className="card mb-8">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-heading font-semibold flex items-center gap-2">
                    <TestTube2 className="w-5 h-5 text-teal" />
                    Robustness Test (JPEG Compression)
                  </h3>
                  <Button
                    onClick={handleRobustnessTest}
                    loading={isTestingRobustness}
                    disabled={isTestingRobustness}
                  >
                    {isTestingRobustness ? 'Testing...' : 'Run Test'}
                  </Button>
                </div>
                
                {robustnessResults && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                  >
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {robustnessResults.map((result) => (
                        <div key={result.quality} className="text-center">
                          <img 
                            src={result.recoveredImage}
                            alt={`Quality ${result.quality}`}
                            className="w-full aspect-square object-cover rounded-lg mb-2"
                          />
                          <p className="font-semibold">JPEG Q{result.quality}</p>
                          <p className={`text-sm ${result.ssim >= 0.8 ? 'text-teal' : result.ssim >= 0.6 ? 'text-amber' : 'text-red-400'}`}>
                            SSIM: {result.ssim.toFixed(4)}
                          </p>
                        </div>
                      ))}
                    </div>
                    <p className="text-sm text-slate-blue mt-4 text-center">
                      Shows how well the secret can be recovered after JPEG compression at different quality levels.
                    </p>
                  </motion.div>
                )}
              </div>

              {/* Download Buttons */}
              <div className="flex flex-wrap justify-center gap-4">
                <Button
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = results!.stegoImage;
                    link.download = 'stego_image.png';
                    link.click();
                  }}
                >
                  <Download className="w-5 h-5" />
                  Download Stego Image
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = results!.recoveredSecret;
                    link.download = 'recovered_secret.png';
                    link.click();
                  }}
                >
                  <Download className="w-5 h-5" />
                  Download Recovered
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleGenerateReport}
                  loading={isGeneratingReport}
                  disabled={isGeneratingReport}
                >
                  <FileText className="w-5 h-5" />
                  {isGeneratingReport ? 'Generating...' : 'Download PDF Report'}
                </Button>
              </div>

              {/* Processing Time */}
              <p className="text-center text-sm text-slate-blue mt-6">
                Processed in {results!.metrics.processingTime.toFixed(2)}s
              </p>
            </>
          )}

          {hasExtractResults && (
            <>
              {/* Extracted Image */}
              <div className="max-w-[300px] mx-auto mb-8">
                <ResultImage
                  label="Extracted Secret"
                  src={extractResults!.recoveredSecret}
                  downloadName="extracted_secret.png"
                />
              </div>

              {/* Download Button */}
              <div className="flex justify-center">
                <Button
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = extractResults!.recoveredSecret;
                    link.download = 'extracted_secret.png';
                    link.click();
                  }}
                >
                  <Download className="w-5 h-5" />
                  Download Extracted Secret
                </Button>
              </div>

              {/* Processing Time */}
              <p className="text-center text-sm text-slate-blue mt-6">
                Extracted in {extractResults!.processingTime.toFixed(2)}s
              </p>
            </>
          )}
        </motion.div>
      </div>
    </section>
  );
};
