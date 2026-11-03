# Supabase setup

1. Apply the SQL migrations to the Supabase project.
2. Create a **private** Storage bucket named `medical-documents` (or use the
   value configured by `SUPABASE_DOCUMENT_BUCKET`).
3. Restrict the bucket to `application/pdf` and set its file-size limit to the
   same value as `MAX_UPLOAD_BYTES`.
4. Configure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` only in the backend
   deployment environment. Never expose the service-role key to a browser or
   commit it to source control.

The API stores objects below
`tenants/{tenant_id}/documents/{document_id}/original.pdf`. The database
function creates the document, ingestion job, outbox event, and audit event in
one transaction.
