/** Typed-by-contract client for AWM-JGEM resources. */

import api from '../services/api';

export async function getAWM(studyId) {
  const response = await api.get(`/studies/${studyId}/awm`);
  return response.data;
}

export async function getSceneGraph(studyId) {
  const response = await api.get(`/studies/${studyId}/scene-graph`);
  return response.data;
}

export async function askGroundedQuestion(studyId, question, options = {}) {
  const response = await api.post(`/studies/${studyId}/ask`, {
    question,
    actor_id: options.actorId || 'local-user',
    actor_role: options.actorRole || 'clinician',
    selected_finding_id: options.selectedFindingId || null,
  });
  return response.data;
}

export async function evaluateIntent(intent) {
  const response = await api.post('/intents/evaluate', intent);
  return response.data;
}

export async function getAudit(studyId) {
  const response = await api.get(`/audit/${studyId}`);
  return response.data;
}
