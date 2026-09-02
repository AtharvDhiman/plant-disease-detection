import { Compass } from 'lucide-react';
import { Link } from 'react-router-dom';

import { Button, Card, EmptyState } from '../components/ui';

export default function NotFound() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-20 sm:px-6">
      <Card>
        <EmptyState
          icon={Compass}
          title="Page not found"
          description="That route does not exist in this application."
          action={
            <Link to="/">
              <Button>Back to the analyser</Button>
            </Link>
          }
        />
      </Card>
    </div>
  );
}
