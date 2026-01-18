interface LoadingBarProps {
  progress: number;
  label?: string;
}

export const LoadingBar = ({ progress, label }: LoadingBarProps) => {
  return (
    <div className="w-full">
      {label && (
        <p className="text-sm text-slate-blue mb-2 text-center">{label}</p>
      )}
      <div className="h-2 bg-navy-900 rounded-full overflow-hidden">
        <div
          className="h-full loading-bar rounded-full transition-all duration-300"
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
      <p className="text-xs text-slate-blue mt-1 text-center">
        {Math.round(progress)}%
      </p>
    </div>
  );
};
