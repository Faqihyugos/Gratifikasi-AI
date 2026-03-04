'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { statusBadge, confidenceBadge } from '@/components/ui/Badge';
import { getRecords } from '@/lib/api';
import { GratifikasiRecord } from '@/types';

export default function MyReportsPage() {
  const [records, setRecords] = useState<GratifikasiRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getRecords({ mine: 1 })
      .then((d) => setRecords(d.results ?? []))
      .catch(() => setError('Failed to load records.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <DashboardLayout title="My Reports">
      <Card>
        {loading && <LoadingSpinner />}
        {error && <p className="text-sm text-red-600">{error}</p>}
        {!loading && !error && records.length === 0 && (
          <p className="text-sm text-gray-500">You have no reports yet.</p>
        )}
        {!loading && records.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">ID</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Description</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Value (IDR)</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Status</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">AI Label</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Confidence</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-500">Date</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 bg-white">
                {records.map((r) => (
                  <tr key={r.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 font-mono text-gray-400">#{r.id}</td>
                    <td className="max-w-xs px-4 py-3 truncate text-gray-900">{r.text}</td>
                    <td className="px-4 py-3 text-gray-700">
                      {r.value_estimation.toLocaleString('id-ID')}
                    </td>
                    <td className="px-4 py-3">{statusBadge(r.status)}</td>
                    <td className="px-4 py-3 text-gray-700">{r.ai_label ?? '—'}</td>
                    <td className="px-4 py-3">
                      {r.ai_confidence != null ? confidenceBadge(r.ai_confidence) : '—'}
                    </td>
                    <td className="px-4 py-3 text-gray-500">
                      {new Date(r.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-3">
                      <Link
                        href={`/case/${r.id}`}
                        className="text-blue-600 hover:underline text-xs font-medium"
                      >
                        View →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </DashboardLayout>
  );
}
