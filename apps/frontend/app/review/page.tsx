'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { confidenceBadge, Badge } from '@/components/ui/Badge';
import { getRecords } from '@/lib/api';
import { GratifikasiRecord } from '@/types';

type SortKey = 'confidence' | 'date' | 'value';

export default function ReviewPage() {
  const [records, setRecords] = useState<GratifikasiRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>('confidence');

  useEffect(() => {
    getRecords({ status: 'WAITING_APPROVAL' })
      .then((d) => setRecords(d.results ?? []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const sorted = [...records].sort((a, b) => {
    if (sortKey === 'confidence') return (b.ai_confidence ?? 0) - (a.ai_confidence ?? 0);
    if (sortKey === 'value') return (b.value_estimation ?? 0) - (a.value_estimation ?? 0);
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

  function isHighRisk(r: GratifikasiRecord) {
    return r.ai_confidence >= 0.85 && r.ai_label === 'Milik Negara';
  }

  return (
    <DashboardLayout title="Review Queue">
      <div className="mb-4 flex items-center justify-between">
        <p className="text-sm text-gray-500">{records.length} cases awaiting review</p>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Sort by:</span>
          {(['confidence', 'date', 'value'] as SortKey[]).map((k) => (
            <button
              key={k}
              onClick={() => setSortKey(k)}
              className={`rounded px-3 py-1 text-xs font-medium capitalize transition-colors ${
                sortKey === k
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-600 border border-gray-300 hover:bg-gray-50'
              }`}
            >
              {k}
            </button>
          ))}
        </div>
      </div>

      <Card>
        {loading && <LoadingSpinner />}
        {!loading && sorted.length === 0 && (
          <p className="text-sm text-gray-500">No cases awaiting review.</p>
        )}
        {!loading && sorted.length > 0 && (
          <div className="space-y-3">
            {sorted.map((r) => (
              <div
                key={r.id}
                className={`rounded-lg border p-4 ${
                  isHighRisk(r)
                    ? 'border-red-300 bg-red-50'
                    : 'border-gray-200 bg-white hover:bg-gray-50'
                }`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-xs text-gray-400">#{r.id}</span>
                      {isHighRisk(r) && (
                        <Badge label="HIGH RISK" variant="danger" />
                      )}
                      <Badge
                        label={r.ai_label ?? 'Pending'}
                        variant={r.ai_label === 'Milik Negara' ? 'danger' : r.ai_label ? 'success' : 'neutral'}
                      />
                    </div>
                    <p className="text-sm text-gray-900 line-clamp-2">{r.text}</p>
                    <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
                      <span>IDR {r.value_estimation?.toLocaleString('id-ID')}</span>
                      <span>{new Date(r.created_at).toLocaleDateString()}</span>
                      {r.relationship && <span className="capitalize">{r.relationship}</span>}
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    {r.ai_confidence != null && confidenceBadge(r.ai_confidence)}
                    <Link
                      href={`/case/${r.id}`}
                      className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
                    >
                      Review →
                    </Link>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </DashboardLayout>
  );
}
