<template>
  <div class="d-sm-flex">
    <div class="ml-1">{{ name }}</div>
    <div class="sadfg" v-for="val in value" :key="val">{{val}}</div>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';

@Component
export default class HelloWorld extends Vue {
  @Prop({ type: [Number] }) value!: [number];

  @Prop({ type: String }) name!: string;
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  ul {
    list-style-type: none;
    padding: 0;
  }

  li {
    display: inline-block;
    margin: 0 10px;
  }

  .sadfg {
    margin: 0 10px;
    padding: 0 1em 0 0.3em;
    background: #42b983;
  }
</style>
