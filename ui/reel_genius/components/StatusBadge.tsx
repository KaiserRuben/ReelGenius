export type TaskStatus = 'completed' | 'running' | 'queued' | 'failed' | string;

interface StatusBadgeProps {
  status: TaskStatus;
  size?: 'sm' | 'md' | 'lg';
}

export default function StatusBadge({ status, size = 'md' }: StatusBadgeProps) {
  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return 'text-xs px-1.5 py-0.5';
      case 'lg':
        return 'text-sm px-3 py-1';
      case 'md':
      default:
        return 'text-xs px-2.5 py-0.5';
    }
  };
  
  let statusClass = '';
  let statusText = status;
  
  switch (status.toLowerCase()) {
    case 'completed':
      statusClass = 'status-completed';
      break;
    case 'running':
      statusClass = 'status-running';
      break;
    case 'queued':
      statusClass = 'status-queued';
      break;
    case 'failed':
      statusClass = 'status-failed';
      break;
    default:
      statusClass = 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
  }
  
  return (
    <span className={`status-badge ${statusClass} ${getSizeClasses()}`}>
      {statusText}
    </span>
  );
}