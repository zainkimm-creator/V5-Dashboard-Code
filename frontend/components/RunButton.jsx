import { LoaderCircle, Play } from 'lucide-react';

export default function RunButton({ children, loading, onClick }) {
  return (
    <button className="primary-button" type="button" onClick={onClick} disabled={loading}>
      {loading ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />}
      <span>{children}</span>
    </button>
  );
}
