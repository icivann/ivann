<template>
  <div class="d-sm-flex">
    <button class="btn btn-primary btn-sm mx-auto" @click="enterCodeVault(option.customNode)">
      Enter Func
    </button>
  </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Mutation } from 'vuex-class';
import Custom from '@/nodes/model/custom/Custom';

@Component
export default class CodeVaultButton extends Vue {
  @Mutation('enterCodeVault') enterCodeVault!: (node?: Custom) => void;
  @Prop() value!: number;

  @Prop({ type: String }) name!: string;

  @Prop({ type: Object })
  option!: {
    customNode?: Custom;
  };
}
</script>
