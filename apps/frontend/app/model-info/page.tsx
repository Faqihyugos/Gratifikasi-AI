'use client';

import React, { useEffect, useState } from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { getModelInfo, triggerRetraining } from '@/lib/api';
import { useToast } from '@/context/ToastContext';
import { ModelInfo } from '@/types';

export default function ModelInfoPage() {
  const { addToast } = useToast();
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch(() => addToast('Failed to load model info.', 'error'))
      .finally(() => setLoading(false));
  }, [addToast]);

  async function handleRetrain() {
    setRetraining(true);
    try {
      const res = await triggerRetraining();
      addToast(res.message ?? 'Retraining triggered successfully.', 'success');
    } catch {
      addToast('Failed to trigger retraining.', 'error');
    } finally {
      setRetraining(false);
    }
  }

  if (loading) {
    return (
      <DashboardLayout title="Model Information">
        <div className="flex justify-center py-12"><LoadingSpinner size="lg" /></div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout title="Model Information">
      <div className="mx-auto max-w-2xl space-y-6">
        <Card
          title="Current Model"
          action={
            <Button
              size="sm"
              variant="primary"
              loading={retraining}
              onClick={handleRetrain}
            >
              🔄 Trigger Retraining
            </Button>
          }
        >
          {!info ? (
            <p className="text-sm text-gray-500">No model information available.</p>
          ) : (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Model Version</span>
                  <div className="font-medium text-gray-900">{info.model_version}</div>
                </div>
                <div>
                  <span className="text-gray-500">Status</span>
                  <div className="font-medium text-gray-900 capitalize">{info.status}</div>
                </div>
                <div>
                  <span className="text-gray-500">Training Date</span>
                  <div className="font-medium text-gray-900">
                    {info.training_date ? new Date(info.training_date).toLocaleDateString() : 'N/A'}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Last Retraining</span>
                  <div className="font-medium text-gray-900">
                    {info.last_retraining ? new Date(info.last_retraining).toLocaleDateString() : 'N/A'}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Dataset Size</span>
                  <div className="font-medium text-gray-900">
                    {(info.dataset_size ?? 0).toLocaleString()} samples
                  </div>
                </div>
              </div>

              <div>
                <div className="mb-1 flex justify-between text-sm">
                  <span className="text-gray-600">F1 Score</span>
                  <span className="font-medium">{((info.f1_score ?? 0) * 100).toFixed(1)}%</span>
                </div>
                <ProgressBar
                  value={(info.f1_score ?? 0) * 100}
                  color={(info.f1_score ?? 0) >= 0.85 ? 'green' : (info.f1_score ?? 0) >= 0.7 ? 'amber' : 'red'}
                />
              </div>

              <div>
                <div className="mb-1 flex justify-between text-sm">
                  <span className="text-gray-600">Accuracy</span>
                  <span className="font-medium">{((info.accuracy ?? 0) * 100).toFixed(1)}%</span>
                </div>
                <ProgressBar
                  value={(info.accuracy ?? 0) * 100}
                  color={(info.accuracy ?? 0) >= 0.85 ? 'green' : (info.accuracy ?? 0) >= 0.7 ? 'amber' : 'red'}
                />
              </div>
            </div>
          )}
        </Card>
      </div>
    </DashboardLayout>
  );
}
