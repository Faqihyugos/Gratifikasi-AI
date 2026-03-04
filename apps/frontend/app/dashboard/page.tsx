'use client';

import React, { useEffect, useState } from 'react';
import { DashboardLayout } from '@/components/layout/DashboardLayout';
import { Card } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { useAuth } from '@/context/AuthContext';
import { getRecords } from '@/lib/api';
import { UserRole } from '@/types';
import Link from 'next/link';

interface Stat {
  label: string;
  value: string | number;
  color: string;
}

export default function DashboardPage() {
  const { user } = useAuth();
  const [stats, setStats] = useState<Stat[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getRecords({ page_size: 100 });
        const records = data.results ?? [];
        const total = data.count;
        const pending = records.filter((r) => r.status === 'PENDING' || r.status === 'WAITING_APPROVAL').length;
        const approved = records.filter((r) => r.status === 'APPROVED').length;
        const rejected = records.filter((r) => r.status === 'REJECTED').length;
        setStats([
          { label: 'Total Records', value: total, color: 'bg-blue-500' },
          { label: 'Pending', value: pending, color: 'bg-amber-500' },
          { label: 'Approved', value: approved, color: 'bg-green-500' },
          { label: 'Rejected', value: rejected, color: 'bg-red-500' },
        ]);
      } catch {
        setStats([]);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const quickLinks = [
    { label: 'Submit New Report', href: '/submit', roles: [UserRole.EMPLOYEE, UserRole.SUPERVISOR, UserRole.ADMIN], color: 'bg-blue-600 hover:bg-blue-700' },
    { label: 'View My Reports', href: '/my-reports', roles: [UserRole.EMPLOYEE, UserRole.SUPERVISOR, UserRole.ADMIN], color: 'bg-slate-600 hover:bg-slate-700' },
    { label: 'Review Queue', href: '/review', roles: [UserRole.COMPLIANCE_OFFICER, UserRole.SUPERVISOR, UserRole.ADMIN], color: 'bg-purple-600 hover:bg-purple-700' },
    { label: 'Analytics', href: '/analytics', roles: [UserRole.COMPLIANCE_OFFICER, UserRole.SUPERVISOR, UserRole.AUDITOR, UserRole.ADMIN], color: 'bg-teal-600 hover:bg-teal-700' },
  ].filter((l) => user && l.roles.includes(user.role));

  return (
    <DashboardLayout title="Dashboard">
      <div className="space-y-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">
            Welcome back, {user?.username}
          </h2>
          <p className="text-sm text-gray-500 capitalize">
            Role: {user?.role.replace('_', ' ')}
          </p>
        </div>

        {loading ? (
          <LoadingSpinner />
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {stats.map((s) => (
              <div key={s.label} className="rounded-lg bg-white border border-gray-200 p-4 shadow-sm">
                <div className={`mb-2 h-1 w-8 rounded ${s.color}`} />
                <div className="text-2xl font-bold text-gray-900">{s.value}</div>
                <div className="text-sm text-gray-500">{s.label}</div>
              </div>
            ))}
          </div>
        )}

        <Card title="Quick Actions">
          <div className="flex flex-wrap gap-3">
            {quickLinks.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-md px-4 py-2 text-sm font-medium text-white transition-colors ${l.color}`}
              >
                {l.label}
              </Link>
            ))}
          </div>
        </Card>
      </div>
    </DashboardLayout>
  );
}
