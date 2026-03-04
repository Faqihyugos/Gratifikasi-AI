'use client';

import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { getAnalytics } from '@/lib/api';
import { AnalyticsData } from '@/types';

const COLORS = ['#2563eb', '#16a34a', '#d97706', '#dc2626'];

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      <div className="text-sm font-medium text-gray-600">{label}</div>
      {sub && <div className="mt-1 text-xs text-gray-400">{sub}</div>}
    </div>
  );
}

export default function AnalyticsPage() {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getAnalytics()
      .then(setData)
      .catch(() => setError('Failed to load analytics data.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <DashboardLayout title="Analytics">
        <div className="flex justify-center py-12"><LoadingSpinner size="lg" /></div>
      </DashboardLayout>
    );
  }

  if (error || !data) {
    return (
      <DashboardLayout title="Analytics">
        <p className="text-sm text-red-600">{error || 'No data available.'}</p>
      </DashboardLayout>
    );
  }

  const labelPieData = [
    { name: 'Milik Negara', value: Math.round(data.milik_negara_pct) },
    { name: 'Bukan Milik Negara', value: Math.round(data.bukan_milik_negara_pct) },
  ];

  const sourcePieData = [
    { name: 'Similarity', value: Math.round(data.similarity_based_pct) },
    { name: 'Classifier', value: Math.round(data.classifier_based_pct) },
  ];

  return (
    <DashboardLayout title="Analytics Dashboard">
      <div className="space-y-6">
        {/* Stats */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard label="Total Submissions" value={data.total_submissions} />
          <StatCard label="Override Rate" value={`${Math.round(data.override_rate * 100)}%`} />
          <StatCard
            label="Avg. Approval Time"
            value={`${(data.avg_approval_time_hours ?? 0).toFixed(1)}h`}
          />
          <StatCard
            label="Milik Negara"
            value={`${Math.round(data.milik_negara_pct)}%`}
            sub="of all submissions"
          />
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <Card title="Label Distribution">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={labelPieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                >
                  {labelPieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => `${v}%`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>

          <Card title="Decision Source">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={sourcePieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                >
                  {sourcePieData.map((_, i) => (
                    <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(v) => `${v}%`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </div>

        {/* Submissions over time */}
        {data.submissions_by_month && data.submissions_by_month.length > 0 && (
          <Card title="Submissions by Month">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={data.submissions_by_month} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        )}
      </div>
    </DashboardLayout>
  );
}
