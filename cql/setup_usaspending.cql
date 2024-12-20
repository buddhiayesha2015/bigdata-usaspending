-- Create Keyspace
CREATE KEYSPACE IF NOT EXISTS usaspending_data
    WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}
     AND durable_writes = true;

-- Use Keyspace
USE usaspending_data;

-- Create Tables
CREATE TABLE IF NOT EXISTS awards(
    award_id             text PRIMARY KEY,
    recipient_name       text,
    start_date           date,
    end_date             date,
    award_amount         float,
    awarding_agency      text,
    awarding_sub_agency  text,
    contract_award_type  text,
    funding_agency       text,
    funding_sub_agency   text,
    description          text,
    last_modified_date   date,
    base_obligation_date date
);

CREATE TABLE IF NOT EXISTS download_history(
    download_id    uuid PRIMARY KEY,
    start_time     timestamp,
    end_time       timestamp,
    number_of_rows int
);

CREATE TABLE IF NOT EXISTS recipient_name_with_geo(
    recipient_name text PRIMARY KEY,
    latitude       float,
    longitude      float
);

CREATE TABLE IF NOT EXISTS awarding_sub_agency_with_geo(
    awarding_sub_agency text PRIMARY KEY,
    latitude            float,
    longitude           float
);

CREATE TABLE IF NOT EXISTS total_award_amount_by_recipient (
    recipient_name      text PRIMARY KEY,
    latitude            float,
    longitude           float,
    total_award_amount  float
);

CREATE TABLE IF NOT EXISTS total_award_amount_by_awarding_sub_agency (
    awarding_sub_agency text PRIMARY KEY,
    latitude            float,
    longitude           float,
    total_award_amount  float
);

CREATE TABLE IF NOT EXISTS awarding_sub_agency_by_recipient (
    awarding_sub_agency text,
    recipient_name       text,
    total_award_amount   float,
    PRIMARY KEY ((awarding_sub_agency), recipient_name)
);

CREATE TABLE IF NOT EXISTS awarding_sub_agency_by_month (
    awarding_sub_agency text,
    month int,
    total_award_amount float,
    PRIMARY KEY ((awarding_sub_agency), month)
);
